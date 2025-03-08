def restore_images(flatten_patches, grid_t, grid_h, grid_w, merge_size, temporal_patch_size, patch_size, channel, data_format):
    grid_h_re = grid_h // merge_size
    grid_w_re = grid_w // merge_size

    restored = flatten_patches.reshape(
        grid_t,
        grid_h_re,
        grid_w_re,
        merge_size,
        merge_size,
        channel,
        temporal_patch_size,
        patch_size,
        patch_size
    )

    restored = restored.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)

    total_time = grid_t * temporal_patch_size
    height = grid_h_re * merge_size * patch_size
    width = grid_w_re * merge_size * patch_size

    restored = restored.reshape(total_time, channel, height, width)

    if data_format == "channels_last":
        restored = restored.transpose(0, 2, 3, 1)

    return restored

def pixel_reshape(image, patch_size, merge_size, temporal_patch_size):
    
    if image.dim() == 3:
        patches = image.unsqueeze(0)
    else:
        patches = image
    if patches.shape[0] == 1:
        patches = patches.repeat(temporal_patch_size, 1, 1, 1)

    channel = patches.shape[1]
    resized_height = patches.shape[2]
    resized_width = patches.shape[3]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    patches = patches.view(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    return flatten_patches, (grid_t, grid_h, grid_w)
