import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

# sig_data_path = r"../data/signal/signal.npy"
# sig_label_path = r"../data/signal/labels.npy"
sig_data_path = r"F:/radar_ws/data/signal/signal.npy"
sig_label_path = r"F:/radar_ws/data/signal/labels.npy"

def read_sig(sig_data_path, sig_label_path):
    sig_data = np.load(sig_data_path)
    sig_label = np.load(sig_label_path)
    return sig_data, sig_label


def generate_gasf_gadf_images(signal, image_size=32):
    # Rescale the signal to [0, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    scaled_signal = (signal - signal_min) / (signal_max - signal_min)

    # Reshape the signal to a 2D array
    reshaped_signal = scaled_signal.reshape(1, -1)

    # Generate Gramian Angular Summation Field (GASF)
    gasf = GramianAngularField(image_size=image_size, method='summation')
    gasf_image = gasf.fit_transform(reshaped_signal)

    # Generate Gramian Angular Difference Field (GADF)
    gadf = GramianAngularField(image_size=image_size, method='difference')
    gadf_image = gadf.fit_transform(reshaped_signal)

    return gasf_image[0], gadf_image[0]


def plot_img(signals, item, img_size=640):
    # item: 0:35999, img_size = 640 / 480/ 1024
    gasf_image, gadf_image = generate_gasf_gadf_images(signals[item], image_size=img_size)
    t = np.linspace(1, 1, 2000)
    t = t.reshape(1, 2000)
    # Plot the original signal, GASF, and GADF
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(t, signals[-1])
    plt.title('Original Signal')

    plt.subplot(1, 3, 2)
    plt.imshow(gasf_image, cmap='viridis', origin='lower')
    plt.title('GASF')

    plt.subplot(1, 3, 3)
    plt.imshow(gadf_image, cmap='viridis', origin='lower')
    plt.title('GADF')


def generate_gasf_gadf_images(save_path, signal,label, item=0, image_size=32):
    # Rescale the signal to [0, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    scaled_signal = (signal - signal_min) / (signal_max - signal_min)

    # Reshape the signal to a 2D array
    reshaped_signal = scaled_signal.reshape(1, -1)

    # Generate Gramian Angular Summation Field (GASF)
    gasf = GramianAngularField(image_size=image_size, method='summation')
    gasf_image = gasf.fit_transform(reshaped_signal)

    # Generate Gramian Angular Difference Field (GADF)
    gadf = GramianAngularField(image_size=image_size, method='difference')
    gadf_image = gadf.fit_transform(reshaped_signal)

    # Save GASF image
    plt.imsave(save_path + '/GASF/' + '{}_{}.png'.format(item,int(label[item])), gasf_image[0], cmap='viridis')

    # Save GADF image
    plt.imsave(save_path + '/GADF/' + '{}_{}.png'.format(item,int(label[item])), gadf_image[0], cmap='viridis')

    # Optionally, you can also display the images
    # plt.imshow(gasf_image[0], cmap='viridis')
    # plt.title('GASF Image')
    # plt.show()

    # plt.imshow(gadf_image[0], cmap='viridis')
    # plt.title('GADF Image')
    # plt.show()
    print("SAVE SUCCESS{}".format(item))
    # return gasf_image[0], gadf_image[0]


sig_data, sig_label = read_sig(sig_data_path=sig_data_path, sig_label_path=sig_label_path)

signal_length = sig_data.shape[0]
labels_length = sig_label.shape
print(labels_length)
print(signal_length)
# save_path = r"../data/imgs"
save_path = r"F:/radar_ws/data/imgs"
labels_= {'A':1, 'B':2, 'C':3}
# img_save(save_path,signals=sig_data, item=0, img_size=640, GASF=True, GADF=True)
for i in range(len(sig_label)):
    generate_gasf_gadf_images(save_path,sig_data[i],label=sig_label,item=i,image_size=640)

