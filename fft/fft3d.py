import numpy as np
import torch
import plotly.graph_objects as go
import os


def get_coords_value(ss):
    indices = torch.argwhere(ss > 0)
    values = ss[ss > 0]
    coords_value = torch.cat([
        indices[:, [0]],      
        values.unsqueeze(1),  
        indices[:, 2:]        
    ], dim=1)

    return coords_value


def analyze_voxel_frequency(coords_value, filter_radius=8, grid_size=64):
    if isinstance(coords_value, torch.Tensor):
        data = coords_value.detach().cpu().numpy()
    else:
        data = np.array(coords_value)

    if data.shape[0] == 0:
        return None, None

    batch_id = data[0, 0]
    mask = data[:, 0] == batch_id
    batch_data = data[mask]

    raw_val = batch_data[:, 1]
    zs = batch_data[:, 2].astype(int)
    ys = batch_data[:, 3].astype(int)
    xs = batch_data[:, 4].astype(int)

    dense_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    valid = (zs < grid_size) & (ys < grid_size) & (xs < grid_size)
    zs, ys, xs, raw_val = zs[valid], ys[valid], xs[valid], raw_val[valid]
    dense_grid[zs, ys, xs] = raw_val

    f = np.fft.fftn(dense_grid)
    fshift = np.fft.fftshift(f) 
    raw_magnitude = np.abs(fshift)
    magnitude_spectrum = 20 * np.log(raw_magnitude + 1)

    cz, cy, cx = grid_size // 2, grid_size // 2, grid_size // 2
    
    z_idx, y_idx, x_idx = np.ogrid[:grid_size, :grid_size, :grid_size]
    dist_sq = (z_idx - cz)**2 + (y_idx - cy)**2 + (x_idx - cx)**2
    
    freq_mask = np.ones((grid_size, grid_size, grid_size), dtype=np.float32)
    freq_mask[dist_sq < filter_radius**2] = 0 

    total_energy = np.sum(raw_magnitude)
    high_freq_energy = np.sum(raw_magnitude * freq_mask)
    
    if total_energy > 1e-6:
        hfer_score = high_freq_energy / total_energy
    else:
        hfer_score = 0.0

    # print(f"Batch {int(batch_id)} | 3D HFER: {hfer_score:.5f}")

    fshift_filtered = fshift * freq_mask

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifftn(f_ishift)
    spatial_high_freq_energy = np.abs(img_back)
    token_hf_intensity = spatial_high_freq_energy[zs, ys, xs]

    if token_hf_intensity.max() > token_hf_intensity.min():
        token_hf_intensity = (token_hf_intensity - token_hf_intensity.min()) / (token_hf_intensity.max() - token_hf_intensity.min())

    spatial_data = {
        "x": xs, "y": ys, "z": zs,
        "raw_val": raw_val,
        "hf_score": token_hf_intensity,
        "batch_id": batch_id
    }

    freq_data = {
        "magnitude": magnitude_spectrum, 
        "raw_magnitude": raw_magnitude, 
        "hfer": hfer_score,           
        "grid_size": grid_size
    }

    return spatial_data, freq_data


def plot_spatial_heatmap(spatial_data, filename="spatial_highfreq.html", filter_radius=8, voxel_size=0.9,bg_color='rgb(200, 200, 200)'):
    import plotly.graph_objects as go
    import numpy as np
    import os
    
    if spatial_data is None: return
    cx = -spatial_data["y"]
    cy = spatial_data["x"]
    cz = spatial_data["z"]
    hf_score = spatial_data["hf_score"]
    raw_vals = spatial_data["raw_val"]
    
    num_voxels = len(cx)
    if num_voxels == 0:
        return

    d = voxel_size / 2.0
    offsets = np.array([
        [-d, -d, -d], [-d, -d,  d], [-d,  d, -d], [-d,  d,  d],
        [ d, -d, -d], [ d, -d,  d], [ d,  d, -d], [ d,  d,  d]
    ])

    faces_template = np.array([
        [0, 1, 5], [0, 5, 4], [2, 3, 1], [2, 1, 0], [6, 7, 3], [6, 3, 2],
        [4, 5, 7], [4, 7, 6], [1, 3, 7], [1, 7, 5], [0, 4, 6], [0, 6, 2]
    ])

    centers = np.stack([cx, cy, cz], axis=1)
    all_vertices = centers[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    x_flat = all_vertices[:, :, 0].flatten()
    y_flat = all_vertices[:, :, 1].flatten()
    z_flat = all_vertices[:, :, 2].flatten()

    base_indices = np.arange(num_voxels) * 8
    all_faces = faces_template[np.newaxis, :, :] + base_indices[:, np.newaxis, np.newaxis]
    i_flat = all_faces[:, :, 0].flatten()
    j_flat = all_faces[:, :, 1].flatten()
    k_flat = all_faces[:, :, 2].flatten()

    intensity_flat = np.repeat(hf_score, 8)
    raw_val_flat = np.repeat(raw_vals, 8)

    bg_color = bg_color

    fig = go.Figure(data=[go.Mesh3d(
        x=x_flat, y=y_flat, z=z_flat,
        i=i_flat, j=j_flat, k=k_flat,
        intensity=intensity_flat,
        colorscale='Jet',
        showscale=True,
        colorbar=dict(
            title=dict(text="High-Freq Energy", font=dict(color='black')),
            tickfont=dict(color='black') 
        ),
        opacity=1.0,
        flatshading=True,
        lighting=dict(
            ambient=1.0,      
            diffuse=0.9,     
            roughness=0.2,  
            specular=0.5,  
            fresnel=0.5    
        ),
        lightposition=dict(x=1000, y=1000, z=2000),
        customdata=raw_val_flat,
        hovertemplate="HF-Score: %{intensity:.4f}<br>RawVal: %{customdata:.2f}<extra></extra>"
    )])


    invisible_axis = dict(
        showgrid=False,      
        showline=False,     
        zeroline=False,    
        showticklabels=False, 
        showbackground=False, 
        title='',           
        visible=False       
    )

    fig.update_layout(
        title=None,          
        width=1200, height=900,
        scene=dict(
            xaxis=invisible_axis,
            yaxis=invisible_axis,
            zaxis=invisible_axis,
            aspectmode='data',
            bgcolor=bg_color
        ),
        paper_bgcolor=bg_color,
        margin=dict(r=0, l=0, b=0, t=0) 
    )

    save_path = os.path.abspath(filename)
    fig.write_html(save_path)
    # print(f"Saved to: {save_path}")


def plot_freq_domain(freq_data, filename="freq_domain_spectrum.html"):

    if freq_data is None: return
    mag = freq_data["magnitude"] 
    grid_size = freq_data["grid_size"]
    z, y, x = np.indices(mag.shape)
    
    threshold = np.percentile(mag, 50) 
    mask = mag > threshold
    
    x_plot = x[mask].flatten()
    y_plot = y[mask].flatten()
    z_plot = z[mask].flatten()
    color_val = mag[mask].flatten()

    color_norm = (color_val - color_val.min()) / (color_val.max() - color_val.min())

    fig = go.Figure(data=[go.Scatter3d(
        x=x_plot, y=y_plot, z=z_plot,
        mode='markers',
        marker=dict(
            size=2,
            color=color_val,
            colorscale='Viridis', 
            opacity=0.3, 
            showscale=True,
            colorbar=dict(title="Log Magnitude"),
        ),
        text=[f"Freq Energy: {v:.2f}" for v in color_val],
        hoverinfo='text'
    )])

    c = grid_size // 2
    fig.add_trace(go.Scatter3d(
        x=[c], y=[c], z=[c],
        mode='markers',
        marker=dict(size=5, color='red', symbol='diamond'),
        name='DC (Center)'
    ))

    fig.update_layout(
        title="3D Frequency Domain Spectrum (Log Magnitude)<br>Center=Low Freq, Outer=High Freq",
        scene=dict(
            xaxis=dict(title='Freq X', gridcolor='#444', showbackground=False),
            yaxis=dict(title='Freq Y', gridcolor='#444', showbackground=False),
            zaxis=dict(title='Freq Z', gridcolor='#444', showbackground=False),
            aspectmode='data',
            bgcolor="rgb(20, 20, 20)"
        ),
        paper_bgcolor="rgb(20, 20, 20)",
        font=dict(color="white"),
        margin=dict(r=0, l=0, b=0, t=50)
    )

    save_path = os.path.abspath(filename)
    fig.write_html(save_path)
    # print(f"saved to: {save_path}")


def process_and_visualize(coords_value, output_dir="./output", filter_radius=8,draw_spatial = True,draw_freq = True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spatial_data, freq_data = analyze_voxel_frequency(coords_value, filter_radius)
    coords_scores = package_to_tensor(spatial_data)
    
    if spatial_data is not None and draw_spatial:
        spatial_filename = os.path.join(output_dir, f"3dfft_spatial_R{filter_radius}.html")
        plot_spatial_heatmap(spatial_data, spatial_filename, filter_radius)

    if freq_data is not None and draw_freq:
        freq_filename = os.path.join(output_dir, f"3dfft_frequency_spectrum.html")
        plot_freq_domain(freq_data, freq_filename)
    
    return coords_scores,freq_data["hfer"] 


def package_to_tensor(spatial_data):

    raw_val = torch.as_tensor(spatial_data["hf_score"], dtype=torch.float32)
    zs = torch.as_tensor(spatial_data["z"], dtype=torch.float32)
    ys = torch.as_tensor(spatial_data["y"], dtype=torch.float32)
    xs = torch.as_tensor(spatial_data["x"], dtype=torch.float32)

    packed_tensor = torch.stack([raw_val, zs, ys, xs], dim=1)
    
    return packed_tensor


def plot_coords_scores_to_html(
    coords_scores, 
    filename="coords_visualization.html", 
    max_points=50000, 
    point_size=2,
    colorscale='Viridis',
    title="Coords Scores Visualization"
):
   
    if isinstance(coords_scores, torch.Tensor):
        coords_scores = coords_scores.detach().cpu().numpy()
        
    N = coords_scores.shape[0]
    
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        data = coords_scores[indices]
    else:
        data = coords_scores
    values = data[:, 0]
    z = data[:, 1]
    y = data[:, 2]
    x = data[:, 3]


    fig = go.Figure(data=[go.Scatter3d(
        x=x,  
        y=y, 
        z=z,  
        mode='markers',
        marker=dict(
            size=point_size,
            color=values,      
            colorscale=colorscale, 
            colorbar=dict(title="Score / Value"),
            opacity=0.8    
        ),

        text=[f"Val: {v:.4f}" for v in values],
        hovertemplate='<b>Value</b>: %{text}<br>' +
                      'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
    )])


    fig.update_layout(
        title=f"{title} (N={min(N, max_points)})",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='data' 
        ),
        margin=dict(r=0, l=0, b=0, t=40)  
    )


    fig.write_html(filename)
    # print(f"Saved {filename}")