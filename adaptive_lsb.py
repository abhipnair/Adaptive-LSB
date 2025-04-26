import os
import cv2
import numpy as np

def calculate_edge_strength(image_array:np):
    """
    Calculate edge strength using a Sobel filter.

    Args: 
        image_array: numpy array of the image.
    """
    gradient_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gradient_x**2 + gradient_y**2)


def embed_data_adaptive(image_path:str, message:bytes, output_path:str):
    """
    Embed secret data in a color image using Adaptive LSB Steganography.
    
    Args:
        image_path: path of the image where data to be embedded.
        message: the message to be embedded.
        output_path: image path where to save the image.
    """
    # Load image in RGB (BGR in OpenCV)
    image = cv2.imread(image_path)
    if image is None:
        # Handle a function of message box here.
        raise ValueError("Image not found or format not supported.")

    # Split image into channels (B, G, R)
    b_channel, g_channel, r_channel = cv2.split(image)

    # Calculate edge strength for each channel
    edge_strength_b = calculate_edge_strength(b_channel)
    edge_strength_g = calculate_edge_strength(g_channel)
    edge_strength_r = calculate_edge_strength(r_channel)

    # Combine edge strengths and normalize to adaptive levels
    combined_edge_strength = (edge_strength_b + edge_strength_g + edge_strength_r) / 3
    adaptive_levels = np.clip((combined_edge_strength / np.max(combined_edge_strength) * 3).astype(int), 1, 3)

    # Convert secret data to binary
    secret_binary = ''.join(format(byte, '08b') for byte in message) + '11111111'  # Unique delimiter
    if len(secret_binary) > image.size * 3:
        # Handle a function of message box here.
        raise ValueError("Image not found or format not supported.")


    # Embed data in all channels (cycling through B, G, R)
    secret_index = 0
    channels = [b_channel, g_channel, r_channel]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if secret_index >= len(secret_binary):
                break

            for channel_index, channel in enumerate(channels):
                if secret_index >= len(secret_binary):
                    break

                # Get adaptive level for the current pixel
                adaptive_level = adaptive_levels[i, j]

                # Modify pixel value
                pixel_value = int(channel[i, j])  # Start with the current channel
                for bit_position in range(adaptive_level):
                    if secret_index >= len(secret_binary):
                        break
                    pixel_value = (pixel_value & ~(1 << bit_position)) | (int(secret_binary[secret_index]) << bit_position)
                    secret_index += 1

                channel[i, j] = np.uint8(np.clip(pixel_value, 0, 255))

    # Merge channels back into an RGB image
    stego_image = cv2.merge((b_channel, g_channel, r_channel))

    # Save the stego image
    cv2.imwrite(output_path, stego_image)
    print(f"Data embedded successfully in {output_path}")


def extract_data_adaptive(stego_image_path:str):
    """
    Extract secret data from a color stego image.
    Args:
        stego_image_path: saved image path.
    """

    if not os.path.exists(stego_image_path):
            print("Image not found.")


    image = cv2.imread(stego_image_path)
    if image is None:
        # Handle a function of message box here.
        raise ValueError("Image not found or format not supported.")

    # Extract channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Calculate edge strength for combined channels
    edge_strength_b = calculate_edge_strength(b_channel)
    edge_strength_g = calculate_edge_strength(g_channel)
    edge_strength_r = calculate_edge_strength(r_channel)
    combined_edge_strength = (edge_strength_b + edge_strength_g + edge_strength_r) / 3

    # Normalize edge strength to adaptive levels
    adaptive_levels = np.clip((combined_edge_strength / np.max(combined_edge_strength) * 3).astype(int), 1, 3)

    # Extract binary data
    secret_binary = ''
    channels = [b_channel, g_channel, r_channel]
    for i in range(b_channel.shape[0]):
        for j in range(b_channel.shape[1]):
            for channel in channels:
                pixel_value = channel[i, j]
                adaptive_level = adaptive_levels[i, j]

                for bit_position in range(adaptive_level):
                    secret_binary += str((pixel_value >> bit_position) & 1)

    # Convert binary to bytes
    byte_array = bytearray()
    for i in range(0, len(secret_binary), 8):
        byte = secret_binary[i:i+8]
        if byte == '11111111':  # Unique delimiter
            break
        byte_array.append(int(byte, 2))

    if not byte_array:
        # Handle a function of message box here.
        raise ValueError("Image not found or format not supported.")

    return bytes(byte_array)



def calculate_difference(original_path, stego_path):
    """
    Calculate absolute pixel-wise difference between original and stego images.
    """
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)

    if original is None or stego is None:
        raise ValueError("Error loading images.")

    difference = cv2.absdiff(original, stego)
    total_difference = np.sum(difference)
    print(f"Total pixel-wise difference: {total_difference}")
    return total_difference


def calculate_difference_percentage(original_path, stego_path):
    """
    Calculate percentage of difference between original and stego images.
    """
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)

    if original is None or stego is None:
        raise ValueError("Error loading images.")

    difference = cv2.absdiff(original, stego)
    total_pixels = original.shape[0] * original.shape[1] * original.shape[2]
    diff_percentage = (np.sum(difference) / (total_pixels * 255)) * 100

    print(f"Percentage difference between original and stego image: {diff_percentage:.6f}%")
    return diff_percentage


def calculate_psnr(original_path, stego_path):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between the original and stego images.
    Higher PSNR indicates better quality.
    """
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)

    if original is None or stego is None:
        raise ValueError("Error loading images.")

    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        psnr = 100  # Perfect match
    else:
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    print(f"PSNR between original and stego image: {psnr:.2f} dB")
    return psnr





# Example Workflow
embed_data_adaptive(
    "assets/sample_img02.jpeg",
    "Hi handsome.".encode("utf-8"),
    "outputs/sample_img01_stegoimage.jpeg"
)



extracted_data = extract_data_adaptive("outputs/sample_img02_stegoimage.jpeg")
print(extracted_data.decode())


# Differences and Robustness Metrics
calculate_difference("assets/sample_img02.jpeg", "outputs/sample_img02_stegoimage.jpeg")
calculate_difference_percentage("assets/sample_img02.jpeg", "outputs/sample_img02_stegoimage.jpeg")
calculate_psnr("assets/sample_img02.jpeg", "outputs/sample_img02_stegoimage.jpeg")

