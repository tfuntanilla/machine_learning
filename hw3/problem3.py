from PIL import Image


def train():
    train_image = 'family.jpg'
    train_mask = 'family.png'

    # Initialize lists for tracking skin_pixels and background_pixels to all 0s
    skin_pixels = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]
    background_pixels = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]

    # Get all skin pixels and background pixels in RGB
    image = Image.open(train_image)
    image = image.convert('RGB')
    mask = Image.open(train_mask)
    mask = mask.convert('RGB')

    width, height = image.size

    # Compare mask to the original
    # If mask pixel value is 255 then it's skin otherwise it's background
    for y in range(height):
        for x in range(width):
            red, green, blue = image.getpixel((x, y))
            mask_red, mask_green, mask_blue = mask.getpixel((x, y))

            if mask_red == 255 and mask_green == 255 and mask_blue == 255:
                skin_pixels[red][green][blue] += 1
            else:
                background_pixels[red][green][blue] += 1
    image.close()
    mask.close()

    # Construct the classifier
    skin = 0
    background = 0

    # Each pixel value (RGB) is mapped to its probability of being a skin color
    for i in range(256):
        for j in range(256):
            for k in range(256):
                skin_pixels[i][j][k] += 1
                skin += skin_pixels[i][j][k]
                background_pixels[i][j][k] += 1
                background += background_pixels[i][j][k]

    skin_probability = skin / (skin + background)
    background_probability = background / (skin + background)
    print('Train P(H0) = ' + str(skin_probability))
    print('Train P(H1) = ' + str(background_probability))

    # Save the classifier
    file = open('classifier', 'w')
    file.write('')
    for i in range(256):
        for j in range(256):
            for k in range(256):
                probability = skin_pixels[i][j][k] * skin_probability / (
                        skin_pixels[i][j][k] + background_pixels[i][j][k])
                file.write(str(probability) + '\n')
    file.close()


def test():
    test_image = 'portrait.jpg'
    threshold = .20

    ratio = [[[0 for k in range(256)] for j in range(256)] for i in range(256)]

    with open('classifier', 'r') as file:
        for i in range(256):
            for j in range(256):
                for k in range(256):
                    probability = file.readline()
                    ratio[i][j][k] = float(probability)

    image = Image.open(test_image)
    image = image.convert('RGB')
    width, height = image.size

    result = Image.new(image.mode, image.size)

    for y in range(height):
        for x in range(width):
            red, green, blue = image.getpixel((x, y))
            if ratio[red][green][blue] > threshold:
                result.putpixel((x, y), (255, 255, 255))
            else:
                result.putpixel((x, y), (0, 0, 0))

    result.save('result.png')


def print_stats():
    # Print stats
    ground_truth = Image.open('portrait.png')
    ground_truth = ground_truth.convert('RGB')
    width, height = ground_truth.size

    true_skin_pixels = 0
    for y in range(height):
        for x in range(width):
            gt_red, gt_green, gt_blue = ground_truth.getpixel((x, y))
            if gt_red == 255 and gt_green == 255 and gt_blue == 255:
                true_skin_pixels += 1

    true_bg_pixels = (width * height) - true_skin_pixels
    print("Total true skin pixels = " + str(true_skin_pixels))
    print("Total true background pixels = " + str(true_bg_pixels))

    predicted = Image.open('result.png')

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for y in range(height):
        for x in range(width):
            gt_red, gt_green, gt_blue = ground_truth.getpixel((x, y))
            pred_red, pred_green, pred_blue = predicted.getpixel((x, y))
            if gt_red == 255 and gt_green == 255 and gt_blue == 255:
                if pred_red == 255 and pred_green == 255 and pred_blue == 255:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if pred_red == 255 and pred_green == 255 and pred_blue == 255:
                    false_positive += 1
                else:
                    true_negative += 1

    tp_rate = true_positive / true_skin_pixels
    tn_rate = true_negative / true_bg_pixels
    fp_rate = false_positive / true_bg_pixels
    fn_rate = false_negative / true_skin_pixels
    print('True positive rate = ' + str(true_positive) + '/' + str(true_skin_pixels) + ' = ' + str(tp_rate))
    print('True negative rate = ' + str(true_negative) + '/' + str(true_bg_pixels) + ' = ' + str(tn_rate))
    print('False positive rate = ' + str(false_positive) + '/' + str(true_bg_pixels) + ' = ' + str(fp_rate))
    print('False negative rate = ' + str(false_negative) + '/' + str(true_skin_pixels) + ' = ' + str(fn_rate))


if __name__ == '__main__':
    train()
    test()
    print_stats()
    print("Done!")
