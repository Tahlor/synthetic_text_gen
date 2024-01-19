from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import cv2
# import img_f as cv2
import numpy as np
import skimage
import string
import random
import os, sys
import math, re, csv
import timeit
import einops

regex_num = re.compile(r'\d')
HEIGHT_RANGE_TEST_STRING='Tlygj|)]'
HEIGHT_RANGE_TEST_STRING_WITH_NUM='Tlygj|1)]'
# import pyvips

# https://stackoverflow.com/a/47269413/1018830
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

class SyntheticWordBase:
    
    def __init__(self, use_cv2_generator=False):
        self.font_dict = {}
        self.use_cv2_generator = use_cv2_generator
        if use_cv2_generator:
            self.ft2 = cv2.freetype.createFreeType2()

    def load_font(self, font_path):
        font_name = Path(font_path).stem
        if font_name in self.font_dict:
            font_obj = self.font_dict[font_name]
        else:
            if self.use_cv2_generator:
                font_obj = self.ft2.loadFontData(fontFileName=font_path, id=0)
            else:
                font_obj = ImageFont.truetype(str(font_path), 100)
            self.font_dict[font_name] = font_obj
        return font_obj

    def render_text(self,  text, font_path, hasNums, hasLower, hasBrackets, min_y, max_y, ink=0.99):
        font = self.load_font(font_path)
        if not hasLower:
            text = text.upper()
        if not hasBrackets:
            text = text.translate(str.maketrans('', '', '()[]'))  # remove brackets
        if not hasNums:
            text = regex_num.sub('', text)
        text = text.strip()
        if text:
            return self.render_pil(font, text, min_y, max_y)
        else:
            return None, None

    def render_pil(self, font, text, minY, maxY):
        np_image = self._render_pil(font, text)
        horzP = np.max(np_image, axis=0)
        minX = first_nonzero(horzP, 0)
        maxX = last_nonzero(horzP, 0)
        if (minX < maxX and minY < maxY):
            # print('original {}'.format(np_image.shape))
            # return np_image,new_text,minX,maxX,minY,maxY,font, f_index,ink
            return np_image[minY:maxY + 1, minX:maxX + 1], text
        else:
            # print('uhoh, blank image, what do I do?')
            return None, None

    def _render_pil(self, font, text):
        for retry in range(7):
            # create big canvas as it's hard to predict how large font will render
            size = (250 + 190 * max(2, len(text)) + 200 * retry, 920 + 200 * retry)
            image = Image.new(mode='L', size=size)

            draw = ImageDraw.Draw(image)
            try:
                draw.text((400, 250), text, font=font, fill=1)
            except OSError:
                print('ERROR: failed generating text "{}"'.format(text))
                continue

        np_image = np.array(image)
        return np_image

    def colorize(self, img, color):
        if isinstance(color, int):
            img[img == 0] = color
        elif isinstance(color, (tuple, list)):
            if len(color) == 1:
                img[img == 0] = color
            elif len(color) == 3:
                img = einops.repeat(img, 'h w -> h w c', c=3)
                img[img[:, :, 0] == 0] = einops.repeat(np.array(color), 'c -> 1 1 c', )
        return img

    def render_cv2(self, font, text: str, font_size: float = 1, thickness: int = 1, color=None) -> np.ndarray:
        """
        TODO: you need to get the font generation working, load in the font, etc.

        Generate an image of text using OpenCV and crop it to the bounding box of the text.

        Args:
            text (str): The text to render.
            font_scale (float): Font scale factor that is multiplied by the font-specific base size.
            thickness (int): Thickness of the lines used to draw the text.

        Returns:
            np.ndarray: Cropped image containing the rendered text.
        """
        # Define font and text properties
        # font = cv2.FONT_HERSHEY_SIMPLEX
        font = self.load_font(font)

        color = (0, 0, 0)  # Black color
        font_scale = self.get_font_scale(font, font_size)

        # Get text size and baseline
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Initialize a white image with dimensions based on text
        image = np.ones((text_height + baseline, text_width, 3), dtype=np.uint8) * 255

        # Draw the text on the image
        cv2.putText(image, text, (0, text_height), font, font_scale, color, thickness)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray_image


class SyntheticWord(SyntheticWordBase):

    def __init__(self, font_dir, clean=True, clear=False, use_cv2_generator=False):
        super().__init__(use_cv2_generator)
        self.font_scale_dict = {}
        self.font_dict = {}

        self.font_dir = Path(font_dir)
        if clean or clear:
            if clear:
                csv_file = 'clear_fonts.csv'
            else:
                csv_file = 'clean_fonts.csv'
            with open(self.font_dir / csv_file) as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                if clear:
                    self.fonts = [(path, True, True, True) for path, case, num in reader]
                    self.bracket_fonts = self.fonts
                else:
                    self.fonts = [(path, case != 'False', num != 'False', bracket != 'False') for
                                  path, case, num, bracket in reader]
                    self.bracket_fonts = [font for font in self.fonts if font[3]]
            self.fonts = self.fonts[1:]  # discard header row

        else:
            with open(self.font_dir / 'fonts.list') as f:
                self.fonts = f.read().splitlines()
        self.create_font_dict()
        self.fonts_with_nums = self.get_fonts_with_nums()

    def get_fonts_with_nums(self):
        return [font for font in self.fonts if font[2]]

    def create_font_dict(self):
        self.font_dict = {}
        for i, (filename, hasLower, hasNums, hasBrackets) in enumerate(self.fonts):
            # self.font_dict[filename] = {"hasLower": hasLower,
            #                             "hasNums": hasNums,
            #                             "hasBrackets": hasBrackets
            #                             }
            self.font_dict[filename] = i
            
            
    def getTestFontImages(self, start):
        ret = []
        texts = ['abcdefg', 'hijklmn', 'opqrst', 'uvwxyz', '12345', '67890', 'ABCDEFG', 'HIJKLMN', 'OPQRST', 'UVWXYZ']
        these_fonts = self.fonts[start:start + 1000]
        for index, (filename, hasLower, hasNums, hasBracket) in enumerate(these_fonts):
            if hasNums and hasLower:
                print('rendering {}/{}'.format(index + start, len(these_fonts) + start), end='\r')
                font = ImageFont.truetype(os.path.join(self.font_dir, filename), 100)
                minY, maxY = self.get_min_max_y_of_font(font, 'Tlygj|')
                font = (font, minY, maxY, True)
                bad = False
                images = []
                for s in texts:
                    image, text = self.getRenderedText(font, s)
                    if image is None:
                        bad = True
                        break
                    images.append((s, image))
                if bad:
                    continue
                ret.append((index + start, filename, images))
        return ret

    def load_font_from_file(self, font_file, hasLower, hasNums, hasBrackets):
        try:
            if self.use_cv2_generator:
                font = self.load_font(filename)
            else:
                font = ImageFont.truetype(str(self.font_dir / font_file), 100)
            test_str = HEIGHT_RANGE_TEST_STRING if not hasNums else HEIGHT_RANGE_TEST_STRING_WITH_NUM
            minY, maxY = self.get_min_max_y_of_font(font, test_str)
            fontR = (font, minY, maxY, hasLower, hasBrackets)
            return fontR
        except OSError:
            print(f"Error loading font {filename}")
            return None

    def get_font_relative_path(self, font_file):
        # this is hacky way to convert a complete file path to just what appears in the index
        match = re.search(r'fonts\\[^\\]+\..*', str(font_file))
        if match:
            font_file = match.group().replace('\\', '/')
        return font_file

    def getFont(self, target=None, random_backup=True):
        while True:
            if target is not None:
                target = self.get_font_relative_path(target)
                font_idx = self.font_dict.get(target)
                if font_idx is None:
                    if random_backup:
                        print("Target font not found, using random font")
                        target = None
                    else:
                        return None, None, None, None
                else:
                    filename, hasLower, hasNums, hasBrackets = self.fonts[font_idx]
            if target is None:
                print("No target, using random font")
                index = np.random.choice(len(self.fonts))
                filename, hasLower, hasNums, hasBrackets = self.fonts[index]
            fontR = self.load_font_from_file(filename, hasLower, hasNums, hasBrackets)
            if fontR is not None:
                break
        if hasNums:
            fontNumsR = fontR
            filenameNums = filename
        else:
            while True:
                indexNums = np.random.choice(len(self.fonts_with_nums))
                filenameNums, hasLower, hasNums, hasBrackets = self.fonts[indexNums]
                fontNumsR = self.load_font_from_file(filenameNums, num_required=hasNums)
                if fontNumsR is not None:
                    break

        return (fontR, filename, fontNumsR, filenameNums)

    def getBracketFont(self):
        while True:
            index = np.random.choice(len(self.bracket_fonts))
            filename, hasLower, hasNums, hasBrackets = self.fonts[index]
            try:
                font = ImageFont.truetype(os.path.join(self.font_dir, filename), 100)

                minY, maxY = self.get_min_max_y_of_font(font, HEIGHT_RANGE_TEST_STRING)
                fontR = (font, minY, maxY, hasLower, hasBrackets)
                break
            except OSError:
                pass
        return fontR

    def getRenderedText(self, fontP, text, ink=0.99):
        if isinstance(fontP, tuple):
            font, minY, maxY, hasLower, hasBrackets = fontP
            if not hasLower:
                text = text.upper()
            if not hasBrackets:
                text = text.translate(str.maketrans('', '', '()[]')) # remove brackets

        else:
            font = fontP
            minY = None
            maxY = None

        if self.use_cv2_generator:
            return self.render_cv2(font, text)
        else:
            return self.render_pil(font, text, minY, maxY)
    

    def render(self):
        if self.use_cv2_generator:
            return self.render_cv2(font, text)
        else:
            return self.render_pil(font, text, minY, maxY)


    def get_min_max_y_of_font(self, font, text):
        img = self._render_pil(font, text)
        vertP = np.max(img, axis=1)
        minY = first_nonzero(vertP, 0).item()
        maxY = last_nonzero(vertP, 0).item()
        return minY, maxY


    def get_font_scale(self, font, font_size: float = 1) -> float:
        """
        Compute the font scale for OpenCV's putText function to achieve a desired text height.

        Args:
            font (int): Font to use.
            font_size (float): Desired font size, used to compute font_scale.

        Returns:
            float: The computed font scale.
        """
        if font in self.font_scale_dict:
            return self.font_scale_dict[font]
        else:
            return compute_font_scale(font, font_size)

    def getBrackets(self, fontP=None, paren=True):
        if fontP is not None:
            font, minY, maxY, hasLower, hasBrackets = fontP
        else:
            hasBrackets = False
        if not hasBrackets:
            fontP = self.getBracketFont()
        open_img, _ = self.getRenderedText(fontP, '(' if paren else '[')
        close_img, _ = self.getRenderedText(fontP, ')' if paren else ']')
        if open_img is None or close_img is None:
            return self.getBrackets(paren=paren)
        return open_img, close_img

    def saveTestSamples(self, output_dir, sample_texts, start=0, end=None):
        """
        Saves test sample images for each font in the specified output directory.

        Args:
        output_dir (str): Directory where the sample images will be saved.
        sample_texts (list): List of texts to be rendered for each font.
        start (int, optional): Starting index of fonts to process. Defaults to 0.
        end (int, optional): Ending index of fonts to process. If None, processes all fonts. Defaults to None.
        """
        if end is None:
            end = len(self.fonts)
        these_fonts = self.fonts[start:end]

        for index, font_info in enumerate(these_fonts):
            filename = font_info[0] if isinstance(font_info, tuple) else font_info
            font_path = os.path.join(self.font_dir, filename)
            try:
                font = ImageFont.truetype(font_path, 100)
            except Exception as e:
                print(f"Error loading font {filename}: {e}")
                continue

            for text in sample_texts:
                image, _ = self.getRenderedText(font, text)
                if image is None or not image:
                    continue

                output_path = os.path.join(output_dir, f"{filename}_{text}.png")
                try:
                    Image.fromarray(image).save(output_path)
                    print(f"Saved image to {output_path}")
                except Exception as e:
                    print(e)
                    print(f"Error saving image to {output_path}")


def compute_font_scale(font, desired_height: int, sample_text: str = "Hg") -> float:
    """
    Compute the font scale for OpenCV's putText function to achieve a desired text height.

    Args:
        desired_height (int): The desired text height in pixels.
        sample_text (str): Sample text to measure.

    Returns:
        float: The computed font scale.
    """
    thickness = 1

    # Measure the height of the sample text rendered with a font_scale of 1
    (_, text_height), _ = cv2.getTextSize(sample_text, font, 1, thickness)

    # Compute the required font_scale
    font_scale = desired_height / text_height

    return font_scale

def parser(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Synthetic Word Generator')
    parser.add_argument('--font_dir', type=str, default='text_fonts', help='directory of fonts')
    parser.add_argument('--text', type=str, default='test', help='text to render')
    parser.add_argument('--use_cv2_font_generator', action='store_true', help='use cv2 font generator')

    if args is not None:
        import shlex
        return parser.parse_args(shlex.split(args))
    else:
        return parser.parse_args()

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    font_folder = "G:/s3/synthetic_data/resources/fonts"
    arg = f"--font_dir {font_folder} --text 'They' --use_cv2_font_generator"
    arg = f"--font_dir {font_folder} --text 'They'"
    args = parser(arg)
    font_dir = args.font_dir
    text = args.text
    sw = SyntheticWord(font_dir,
                       clean=True,
                       use_cv2_generator=args.use_cv2_font_generator)
    font, name, fontN, nameN = sw.getFont('fonts/3rd Man.ttf')
    for text in [text]:  # ,text+'y',text+'t']:
        if re.match('\d', text):
            im, text = sw.getRenderedText(fontN, text)
        else:
            im, text = sw.getRenderedText(font, text)
        print(text)

        Image.fromarray(im*255).show()

        # cv2.imshow('x', im)
        # cv2.show()
