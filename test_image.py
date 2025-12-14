from PIL import Image
import matplotlib.pyplot as plt
img=Image.open("image_000035.JPG")
plt.imshow(img)
plt.axis("off")
plt.show()