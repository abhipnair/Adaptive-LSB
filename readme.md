# Adaptive LSB Steganography

## 📝 Overview

This project implements **Adaptive Least Significant Bit (LSB) Steganography** for hiding secret messages inside color images using a more secure edge-detection-based adaptive approach.

It combines:
- Edge detection (Sobel filter)
- Adaptive bit-level embedding
- Secure data extraction
- Robustness verification (PSNR, pixel differences)

Resulting in a highly invisible and resilient steganographic communication technique.

---

## 📅 Features

- 🔐 **Adaptive LSB Embedding**
- 🛡️ **Sobel Filter Edge Strength Calculation**
- 📊 **High PSNR Guarantee** (Over 100 dB)
- 📑 **Secret Message Extraction**
- 📊 **Robustness Analysis:**
  - Pixel-wise Difference
  - Percentage Difference
  - PSNR (Peak Signal-to-Noise Ratio)

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- Numpy

Install dependencies:
```bash
pip install opencv-python numpy
```

---

## 🔄 Usage

### Embed Secret Message
```python
embed_data_adaptive(
    "assets/sample_img01.jpg",
    "Hi handsome.".encode("utf-8"),
    "outputs/sample_img01_stegoimage.png"
)
```

### Extract Hidden Message
```python
extracted_data = extract_data_adaptive("outputs/sample_img01_stegoimage.png")
print(extracted_data.decode())
```

### Calculate Differences and Robustness Metrics
```python
calculate_difference("assets/sample_img01.jpg", "outputs/sample_img01_stegoimage.png")
calculate_difference_percentage("assets/sample_img01.jpg", "outputs/sample_img01_stegoimage.png")
calculate_psnr("assets/sample_img01.jpg", "outputs/sample_img01_stegoimage.png")
```

---

## 🔸 How It Works

1. Load the color image and split into RGB channels.
2. Calculate edge strength using the Sobel operator.
3. Normalize edges into adaptive levels (1-3 LSBs).
4. Embed secret message bits into pixels adaptively.
5. Save the stego image.
6. Extraction reverses the process, recovering the hidden message.

---

## 👩‍💻 Example Output

```
Data embedded successfully in outputs/sample_img01_stegoimage.png
Extracted Message: Hi handsome.
Total pixel-wise difference: 44
Percentage difference: 0.000001%
PSNR between original and stego image: 104.43 dB
```

---

## 📈 Metrics Interpretation

| Metric | Ideal Value | Your Value | Meaning |
|:------|:-----------|:----------|:--------|
| Pixel Difference | Low | 44 | Very minimal visible change |
| Difference % | Very Low (<0.001%) | 0.000001% | Invisible to human eye |
| PSNR | > 40 dB | 104.43 dB | Extremely high visual quality |

---

## 🔐 Future Enhancements

- Add password-based encryption before embedding.
- Expand to video, audio, and document carriers.
- Robustness testing under image compression.

---

## 📅 License

This project is open-source and available under the MIT License.

---

## 📄 References

- Yang, Sun – "Adaptive LSB Substitution for Image Data Hiding"
- Roy, Islam – "Hybrid Secured LSB and AES Approach"
- Cheddad et al. – "Survey of Image Steganography Methods"

---
