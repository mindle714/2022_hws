import imageio as iio
import matplotlib.pyplot as plt

img = iio.imread("lena.png")
bit_planes = []
for i in range(8):
  bit_planes.append((img>>i) % 2)

fig = plt.figure(figsize=(19.2, 4.8))

i=0
for bit_plane in bit_planes[::-1]:
  ax = fig.add_subplot(1, len(bit_planes), i+1)
  ax.set_axis_off()
  #iio.imwrite("bit{}.png".format(i), bit_plane*255)
  ax.imshow(bit_plane * 255, cmap='gray')
  i += 1

#print(img)
#plt.axis('off')
plt.savefig('lena_gray.png')
