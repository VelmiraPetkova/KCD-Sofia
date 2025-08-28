import qrcode

# Example URL
url = "https://www.kubeflow.org/docs/started/"

# Create qr code instance
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# Add data
qr.add_data(url)
qr.make(fit=True)

# Create an image
img = qr.make_image(fill_color="black", back_color="white")

# Save it to file
img.save("my_qr_code.png")

print("QR code saved to my_qr_code.png")
