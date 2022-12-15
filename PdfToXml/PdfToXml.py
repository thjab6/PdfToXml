from calendar import c
import pytesseract
import fitz
from PIL import Image
import io
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError, 
    PDFPageCountError, 
    PDFSyntaxError
)
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
import subprocess
import numpy as np
from collections import Counter
import statistics
from functools import reduce


class DocumentElement:
    """
    Represents an element in the document
    """
    def __init__(self, level, left, top, width, height, text='', style='', type='', info='', align=''):
        self.level = level
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.text = text
        self.style = style
        self.type = type
        self.info = info
        self.align = align

    def __repr__(self):
        return "DocumentElement(" + (
            'type=' + str(self.level) + ', ' +
            'left=' + str(self.left) + ', ' +
            'top=' + str(self.top) + ', ' +
            'width=' + str(self.width) + ', ' +
            'height=' + str(self.height) + ', ' +
            'text=' + str(self.text) + ', ' + 
            'style=' + str(self.style) + ', ' + 
            'type=' + str(self.type)
        ) + ')'

    def convert_to_alto_tag(self, tag_id):
        """
        Returns an alto tag format string
        """
        alto_types = ['Page', 'PrintSpace', 'ComposedBlock', 'TextBlock', 'TextLine', 'String']
        """
        str_without_quote = str(self.text).replace("\"", "")
        return  (f'<{alto_types[self.level]}' 
                f'ID="{str(tag_id)}" '
                f'HPOS ="{str(self.left)}" '
                f'VPOS="{str(self.top)}" '
                f'WIDTH="{str(self.width)}" '
                f'HEIGHT="{str(self.height)}" '
                
                ((f' CONTENT=" {str_without_quote}" ') if self.level == 5 else f'')
                ((f' STYLE="{str(self.style)}"') if self.style != '' and self.level == 5 else f'')
                ((f' TYPE="{str(self.type)}"') if self.type != '' else f'')
                ((f' INFO="{str(self.info)}"') if self.info != '' else f'')
                ((f' ALIGN="{str(self.align)}"') if self.align != '' else f''))
                ('/>' if self.level == 5 else '>'))"""
                
        return '<' + alto_types[self.level] + ' ' + (
            'ID="' + str(tag_id) + '" ' +
            'HPOS="' + str(self.left) + '" ' +
            'VPOS="' + str(self.top) + '" ' +
            'WIDTH="' + str(self.width) + '" ' +
            'HEIGHT="' + str(self.height) + '"' +
            ((' CONTENT="' + str(self.text).replace('"', '') + '"') if self.level == 5 else '') +
            ((' STYLE="' + str(self.style) + '"') if self.style != '' and self.level == 5 else '') +
            ((' TYPE="' + str(self.type) + '"') if self.type != '' else '') +
            ((' INFO="' + str(self.info) + '"') if self.info != '' else '') +
            ((' ALIGN="' + str(self.align) + '"') if self.align != '' else '') +
            ('/' if self.level == 5 else '') +
            '>'
            )


class BlockReordering:
    def __init__(self, parent=None):
        self.elements = []

    def set_elements(self, elements):
        self.elements = elements

    def run(self):
        prev_ind = 0
        new_elements = []
        for i, element in enumerate(self.elements):
            if element.level == 1:
                prev_ind = i
            elif element.level == 2:
                if self.elements[prev_ind].level != 2:
                    prev_ind = i
                elif self.elements[prev_ind].level == 2:
                    distance = element.top - (self.elements[prev_ind].top+self.elements[prev_ind].height)
                    if distance < 5:
                        self.elements[i].info = 'del'
                        self.elements[prev_ind].height += (distance+element.height) 
                        if element.left < self.elements[prev_ind].left:
                            self.elements[prev_ind].width = self.elements[prev_ind].left - element.left
                            self.elements[prev_ind].left = element.left
                        if element.left+element.width > self.elements[prev_ind].left+self.elements[prev_ind].width:
                            self.elements[prev_ind].width = element.left+element.width - self.elements[prev_ind].left
                    else:
                        prev_ind = i

            if self.elements[i].info != 'del':
                new_elements.append(self.elements[i])    

        return new_elements       


def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


class BlockAxis:
    def __init__(self, parent=None):
        self.elements = []

    def set_elements(self, elements):
        self.elements = elements

    def __print_data(self, last_composed, lefts, rights, mids):
        if last_composed is not None:
            print('block left:', last_composed.left)
            print('block right:', last_composed.left+last_composed.width)
            print('block mid:', (2*last_composed.left+last_composed.width)/2)
        print('lefts: ', lefts)
        print('rights:', rights)
        print('mids:  ', mids)
        statistics.stdev(lefts) if len(lefts) >= 2 else lefts[0]
        print(statistics.stdev(lefts) if len(lefts) >= 2 else 0, Average(lefts))
        print(statistics.stdev(rights) if len(rights) >= 2 else 0, Average(rights))
        print(statistics.stdev(mids) if len(mids) >= 2 else 0, Average(mids))
        print('\n')

    def run(self):
        print('blockAcis')
        lefts=[]
        rights=[]
        mids=[]
        last_composed = None
        last_i = 0
        for i, element in enumerate(self.elements):
            
            if element.level==2 and len(lefts)>0:
                #self.__print_data(last_composed, lefts, rights, mids)
                
                mini = statistics.stdev(lefts) if len(lefts) >= 2 else 0
                align = 'left'
                if (statistics.stdev(rights) if len(rights) >= 2 else 0) < mini:
                    mini = statistics.stdev(rights) if len(rights) >= 2 else 0
                    align = 'right'
                
                if (statistics.stdev(mids) if len(mids) >= 2 else 0) < mini:
                    mini = statistics.stdev(mids) if len(mids) >= 2 else 0
                    align = 'mid'

                self.elements[last_i].align = align

                lefts=[]
                rights=[]
                mids=[]

                last_composed = element
                last_i = i
            elif element.level == 4:
                lefts.append(element.left)
                rights.append(element.left+element.width)
                mids.append((2*element.left+element.width)/2)
            if element.level == 2:
                last_composed = element
                last_i = i
        #self.__print_data(last_composed, lefts, rights, mids)
        print('blockAcis end')

        return self.elements

                            
                    

class DocumentClustering:
    def __init__(self, parent=None):
        self.elements = []

    def set_elements(self, elements):
        self.elements = elements

    def run(self):
        #print('dokumentum klaszterezés')
        DOCUMENT_WIDTH = self.elements[0].width
        page_count = len(self.elements)
        composedblock_count = 0
        line_width = []
        line_count = 0
        composedblocks_size = []
        composedblocks_width = []
        composedblocks_lines = []

        for i, element in enumerate(self.elements):
            if element.type != 'pagenumber':
                if element.level == 4: #TextLine
                    line_count += 1
                    line_width.append(element.width)
                if element.level == 2:
                    composedblock_count += 1
                    composedblocks_size.append(0)
                    composedblocks_width.append(element.width / DOCUMENT_WIDTH)
                    composedblocks_lines.append(0)
                if element.level == 4:
                    composedblocks_lines[len(composedblocks_lines)-1] += 1

                if element.level == 5:
                    composedblocks_size[len(composedblocks_size)-1] += len(element.text)+1

        #print('number of composed blocks:', composedblock_count)

        data_frame = pd.DataFrame({
            'composedblocks_size' : composedblocks_size, 
            'composedblocks_width' : composedblocks_width, 
            'composedblocks_lines' : composedblocks_lines
        })

        #sns.scatterplot(x='composedblocks_size', y='composedblocks_width', data=data_frame)
        #print('mentés')
        #plt.savefig('plt/document_size.png')
        #plt.close()

        #sns.scatterplot(x='composedblocks_lines', y='composedblocks_width', data=data_frame)
        #print('mentés')
        #plt.savefig('plt/document_lines.png')
        #plt.close()
        


class LayoutClustering:
    def __init__(self, parent=None):
        self.elements = []

    def set_elements(self, elements):
        """
        Sets the list of DocumentElements
        :param elements: list[DocumentElement]
        """
        self.elements = elements

    def run(self, clustering_type='HIERARCHICAL'):
        linecount_list = []
        W = self.elements[0].width
        H = self.elements[0].height
        letter_count_list = []
        letter_count_number_list = []
        is_upper_list = []
        number_count_list = []
        distance_from_last = []
        distance_from_mid_w = []
        distance_from_mid_h = []
        size_ratio = []
        last = 0
        mid_w = W/2
        mid_h = H/2
        page_size = W*H
        for i, element in enumerate(self.elements):
            
            if element.level == 1:
                last = 0

            if element.level == 2:
                linecount_list.append(0)
                letter_count_list.append(0)
                is_upper_list.append(0)
                number_count_list.append(0)
                letter_count_number_list.append(0)
                distance_from_last.append( ((element.top-last) if (element.top-last) >= 0 else 0 )/H)

                distance_from_mid_w.append(min(1,abs(mid_w-((element.left+element.width/2)))/mid_w))
                distance_from_mid_h.append(min(1,abs(mid_h-((element.top+element.height/2)))/mid_h))
                size_ratio.append((element.width*element.height)/(page_size))

                last = element.top+element.height
            
            if element.level == 4:
                linecount_list[len(linecount_list)-1] += 1
            if element.level == 5:
                letter_count_list[len(letter_count_list)-1] += len(element.text)
                is_upper_list[len(is_upper_list)-1] += sum(1 for c in element.text if c.isupper())
                tmp_text = element.text.replace(' ', '').replace('.', '')
                number_count_list[len(number_count_list)-1] += sum(1 for c in tmp_text if c.isnumeric())
                letter_count_number_list[len(letter_count_number_list)-1] += len(tmp_text)
        
        upper_letter_ratio = [(0 if letter_count_list[i] == 0 else  is_upper_list[i]/letter_count_list[i])  for i in range(len(letter_count_list))]
        number_ratio = [(0 if letter_count_number_list[i] == 0 else  number_count_list[i]/letter_count_number_list[i])  for i in range(len(letter_count_number_list))]

        upper_letter_ratio.append(0.95)
        number_ratio.append(0)

        data_frame2 = pd.DataFrame({
            'upper_letter_ratio' : upper_letter_ratio, 
            'number_ratio' : number_ratio,
        })        
        
        distance_from_mid_w.append(0.8)
        distance_from_mid_h.append(0.8)
        size_ratio.append(0)

        #pd.set_option('max_columns', 200) 
        data_frame = pd.DataFrame({
            'distance_from_mid_w': distance_from_mid_w,
            'distance_from_mid_h': distance_from_mid_h,
            'size_ratio': size_ratio,
        })
        
        if clustering_type == 'HIERARCHICAL':
            Z = linkage(data_frame, 'ward')
            data_frame['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')
        elif clustering_type == 'KMEANS':
            startpts4 = np.array([[0.95, 0.95, 0],
                                  [0.75, 0.8, 0],
                                  [0.75, 0.9, 0],
                                  [0.9, 0.8, 0],
                                  [0, 0, 1],
                                  [0.5, 0.5, 0.5]])
            km = KMeans(n_clusters=6, random_state=0)
            km.fit(data_frame)
            labs = km.fit_predict(data_frame)

            startpts = np.array([[1, 0], [0, 0], [0, 1]])
            km2 = KMeans(n_clusters=3, init=startpts)
            km2.fit(data_frame2)
            data_frame2['cluster_labels'] = km2.predict(data_frame2)

            print('#######################################')
            print('labels:',labs)
            dbs = davies_bouldin_score(data_frame, labs)
            sil = silhouette_score(data_frame, labs)
            cal = calinski_harabasz_score(data_frame, labs)
            print(f'davies_bouldin_score: {dbs}')
            print(f'silhouette: {sil}')
            print(f'cal: {cal}')
            data_frame['cluster_labels'] = km.predict(data_frame)
            print('df:', data_frame)
        elif clustering_type == 'DBSCAN':
            bandwidth = estimate_bandwidth(data_frame, quantile=0.2, n_samples=500)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(data_frame)
            data_frame['cluster_labels'] = ms.labels_

        #try:
        #    plt.close()
        #except:
        #    pass

        #data_name = input('save data frame:')
        #data_frame.to_csv(data_name, encoding='utf-8')

        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('distance_from_mid_w')
        # ax.set_ylabel('distance_from_mid_h')
        # ax.set_zlabel('size_ratio')
        # fg = ax.scatter3D(data_frame['distance_from_mid_w'], data_frame['distance_from_mid_h'], data_frame['size_ratio'], c=data_frame['cluster_labels'])
        # plt.show()


        number_cluster = data_frame['cluster_labels'][len(data_frame['cluster_labels'])-1]
        title_cluster = data_frame2['cluster_labels'][len(data_frame2['cluster_labels'])-1]

        ind = 0
        #cluster_names = ['paragraph', 'pagenumber','none']
        for i in range(len(self.elements)):
            if self.elements[i].level == 2:
                if data_frame['cluster_labels'][ind] == number_cluster:
                    self.elements[i].type = str('pagenumber') #cluster_names[name_ind]
                elif data_frame2['cluster_labels'][ind] == title_cluster:
                    self.elements[i].type = str('title')
                else:
                    self.elements[i].type = str('paragraph') #cluster_names[name_ind]

                ind += 1

        return self.elements


class PoemAnalyzer:
    def __init__(self, parent=None):
        self.elements = []

    def set_elements(self, elements):
        """
        Sets the list of DocumentElements
        :param elements: list[DocumentElement]
        """
        self.elements = elements

    def run(self):
        HUN_VOWELS = "aáeéiíoóöőuúüűAÁEÉIÍOÓÖŐUÚÜŰ"

        syllables = []
        letters = []
        rhymes = []

        count_of_vowels = 0
        last_vowel = ''
        ind = 0
        last = ''

        first_paragraph = True
        last_paragraph_ind = 0

        for i, element in enumerate(self.elements):
            if element.type == 'paragraph':
                if first_paragraph:
                    first_paragraph = False
                    last_paragraph_ind = i
                    continue

                #if len(letters)>0:
                if count_of_vowels>0:
                    syllables.append(count_of_vowels)
                    if last in letters:
                        rhymes.append(letters.index(last))
                    else:
                        letters.append(last)
                        rhymes.append(len(letters)-1)
                count_of_vowels = 0
                last_vowel = ''
                ind = 0
                last = ''

                #print('rím képlet', rhymes)
                if len(rhymes)>0:
                    #pass
                    rhyme_ratio = sum(list(filter(lambda x: (x>1), Counter(rhymes).values())))/len(rhymes)
                    #print('rímelő sorok aránya', rhyme_ratio)
                    if rhyme_ratio >= 0.4:
                        self.elements[last_paragraph_ind].type = 'poem'
                #print('sorvégek', letters)
                #print('szótagok száma', syllables)
                #print('\n')

                syllables = []
                letters = []
                rhymes = []
                last_paragraph_ind = i
            if element.level == 4 or element.level == 3:
                if count_of_vowels > 0:
                    syllables.append(count_of_vowels)
                    if last in letters:
                        rhymes.append(letters.index(last))
                    else:
                        letters.append(last)
                        rhymes.append(len(letters)-1)

                count_of_vowels = 0
                last_vowel = ''
                ind = 0
                last = ''
            if element.level == 5:
                for j, char in enumerate(element.text):
                    if char in HUN_VOWELS:
                        count_of_vowels += 1
                        last_vowel = char
                        last = element.text[j:].replace('\n', '').replace('.', '').replace(', ', '').replace(' ', '').replace('-', '').replace('!', '').replace('?', '').replace(';', '')
                        ind = j
        
        if count_of_vowels>0:
            syllables.append(count_of_vowels)
            if last in letters:
                rhymes.append(letters.index(last))
            else:
                letters.append(last)
                rhymes.append(len(letters)-1)
        count_of_vowels = 0
        last_vowel = ''
        ind = 0
        last = ''

        if len(rhymes)>0:
            rhyme_ratio = sum(list(filter(lambda x: (x>1), Counter(rhymes).values())))/len(rhymes)
            if rhyme_ratio >= 0.4:
                self.elements[last_paragraph_ind].type = 'poem'



class ConvertPdfToImagesI:
    """
    Interface for converting pdf pages to images
    """
    def __init__(self, parent=None):
        self.images = None
        self.parent = parent
        self.images = []
        self.progress = 0

    def convert(self, path):
        pass

    def get_images(self):
        pass

    def get_count(self):
        """
        Returns the number of images
        """
        return len(self.images)

    def get_progress(self):
        """
        returns the status of the processing (how many pages are done)
        """
        return self.progress


class PdfToImgModule(ConvertPdfToImagesI):
    """
    Class == which converts the pages of a pdf into images
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = None
        self.parent = parent
        self.__pageCount = 0

    def convert(self, path):
        """
        Converts the pdf in the specified path to images
        """
        self.images = convert_from_path(path)
        return self.images

    def get_images(self):
        return self.images

class FastImgLoad(ConvertPdfToImagesI):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__pageCount = 0
    
    def get_count(self):
        """
        Returns the number of pages
        """
        return self.__pageCount
    
    def convert(self, path):
        self.progress = 0
        self.images = []
        pdf_file = fitz.open(path)

        for page in pdf_file:
            #zoom = 2    # zoom factor
            #mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(dpi=300)
            pix1 = fitz.Pixmap(pix, 0) if pix.alpha else pix
            img = pix1.tobytes("ppm")#getImageData("ppm")
            image = Image.open(io.BytesIO(img))
            self.images.append(image)

        return self.images

class PyMuPdfModule(ConvertPdfToImagesI):
    """
    Class for extracting images contained in the pdf
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__pageCount = 0

    def get_count(self):
        """
        Returns the number of pages
        """
        return self.__pageCount

    def convert(self, path):
        """
        Converts pages of a pdf to images
        :param path: str, path of the pdf
        :returns: list[Image], list of PIL images
        """
        self.progress = 0
        self.images = []
        pdf_file = fitz.open(path)
        self.__pageCount = len(pdf_file)
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            # image_list = page.get_images()

            for image_index, img in enumerate(page.get_images(), start=1):
                xref = img[0]

                base_image = pdf_file.extract_image(xref)
                # image_bytes = base_image['image']

                image = Image.open(io.BytesIO(base_image['image']))
                self.images.append(image)
            self.progress += 1
            if self.parent is not None:
                try:
                    self.parent.page_processed(self.get_count(), self.progress)
                except Exception as e:
                    print(e)
        return self.images

    def get_images(self):
        return self.images


class TesseractTextOnly:
    def __init__(self, parent=None):
        self.images = None
        self.elements = []
        self.parent = parent
        self.progress = 0
        self.pageCount = 0
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()

    def set_images(self, images: list):
        """
        Set the images list of images
        :param images: list of PIL images
        """
        self.images = images

    def get_count(self):
        """
        :returns: int, the number of images
        """
        return len(self.images)

    def get_progress(self):
        """
        :returns: int, how many pages are done
        """
        return self.progress

    def set_elements(self, elements):
        """
        Sets the list of DocumentElements
        :param elements: list[DocumentElement]
        """
        self.elements = elements

    def run(self):
        new_elements = []
        page_counter = 0

        originalw = self.elements[0].width
        originalh = self.elements[0].height

        neww, newh = self.images[0].size

        multw = neww/originalw
        multh = newh/originalh

        # for i in range(len(self.elements)):
        for i, element in enumerate(self.elements):
            # new_elements.append(element)
            if element.level == 1:
                page_counter += 1
                self.progress += 1
                if self.parent is not None:
                    try:
                        self.parent.page_processed(self.get_count(), self.progress)
                    except Exception as e:
                        print(e)
                new_elements.append(element)
            elif element.level == 5:
                mar = 0
                box = (int(element.left*multw)-mar, 
                       int(element.top*multh)-mar, 
                       int(element.left+element.width)*multw+mar, 
                       int(element.top+element.height)*multh+mar)
                # print(box)
                img = self.images[page_counter-1]
                cropped = img.crop(box)
                # cropped.show()

                data = pytesseract.image_to_string(cropped, lang='hun')
                # print(data)
                # new_elements[i].text = data
                new_elements.append(DocumentElement(element.level, 
                                                    int(element.left*multw), 
                                                    int(element.top*multh), 
                                                    (int(element.left+element.width)*multw), 
                                                    (int(element.top+element.height)*multh), 
                                                    str(data)
                                                    )
                                    )

        return new_elements


class TesseractModule:
    """
    Wrapper class for pytesseract
    Runs tesseract OCR on a list of PIL images, and returns a list of DocumentElements that represent the document
    """
    def __init__(self, parent=None):
        self.images = None
        self.elements = []
        self.parent = parent
        self.progress = 0
        self.pageCount = 0
        self.lock = threading.Lock()
        self.lock2 = threading.Lock()
        self.startTime = time.time()
        self.timesFromStart = []
        self.times = []
        self.__futures = []
        self.__executor = None
        self.__is_shutDown = False

    def set_images(self, images: list):
        """
        Set the images list of images
        :param images: list of PIL images
        """
        self.images = images

    def get_count(self):
        """
        :returns: int, the number of images
        """
        return len(self.images)

    def get_progress(self):
        """
        :returns: int, how many pages are done
        """
        return self.progress

    def run(self):
        """
        Process the images and returns a list of DocumentElements
        :returns: a list of DocumentElement
        """
        # time
        self.startTime = time.time()
        self.timesFromStart = []

        for i in range(len(self.images)):
            self.timesFromStart.append(0)

        self.elements = []
        for i in range(len(self.images)):
            data = pytesseract.image_to_data(self.images[i], 
                                             lang='hun', 
                                             output_type=pytesseract.Output.DICT)

            """ for j in range(len(data['level'])):
                self.elements.append(
                    DocumentElement(data['level'][j], 
                                    data['left'][j], 
                                    data['top'][j], 
                                    data['width'][j], 
                                    data['height'][j], 
                                    data['text'][j])
                ) """

            self.elements += [DocumentElement(data['level'][j], 
                                             data['left'][j], 
                                             data['top'][j], 
                                             data['width'][j], 
                                             data['height'][j], 
                                             data['text'][j]) 
                                            for j in range(len(data['level']))]
            
            self.progress += 1
            # time
            end_time = time.time()
            delta_time_from_start = end_time-self.startTime
            self.timesFromStart[i] = round(delta_time_from_start, 4)

            if self.parent is not None:
                try:
                    self.parent.page_processed(self.get_count(), self.progress)
                except Exception as e:
                    print(e)

        # file_ocr = open('ocr_time.txt', 'w')
        # for i in range(len(self.timesFromStart)):
        #    file_ocr.write(str(self.timesFromStart[i])+' ')

        # file_ocr.close()
        # print('timesFromStart: ', self.timesFromStart)

        return self.elements

    def get_page(self, page_number: int):
        """
        :param: page_number
        :returns: a processed page info
        """

        elements = []
        if page_number > len(self.images):
            print('invalid page number')
            # TODO exception

        data = pytesseract.image_to_data(self.images[page_number], 
                                         lang='hun', 
                                         output_type=pytesseract.Output.DICT)
        """ 
        for j in range(len(data['level'])):
            elements.append(
                DocumentElement(data['level'][j], 
                                data['left'][j], 
                                data['top'][j], 
                                data['width'][j], 
                                data['height'][j], 
                                data['text'][j])
            )"""

        elements = [DocumentElement(data['level'][j],
                                    data['left'][j],
                                    data['top'][j],
                                    data['width'][j],
                                    data['height'][j],
                                    data['text'][j])
                    for j in range(len(data['level']))]

        return elements

    def get_page_plain_text(self, page_number: int):
        """
        :param: page_number
        :returns: a processed page info
        """
        return pytesseract.image_to_string(self.images[page_number],
                                           lang='hun')

    def run_with_threading(self):
        """
        Run Tesseract OCR on the given images, on multiple threads using threading.Thread()
        :returns: a list of DocumentElement
        """
        cpu_count = os.cpu_count()

        threads = []
        page_count = int(len(self.images)/cpu_count)
        pages = [0]
        elements_list = []

        for i in range(cpu_count):
            elements_list.append([])

        for i in range(cpu_count-1):
            pages.append(page_count*(i+1))

        pages.append(len(self.images))

        for i in range(cpu_count):
            threads.append(threading.Thread(target=self.__task_for_threads,
                                            args=[elements_list[i],
                                                  pages[i],
                                                  pages[i+1]]))

        for i in range(cpu_count):
            threads[i].start()

        for i in range(cpu_count):
            threads[i].join()

        for i in range(cpu_count):
            for j in range(len(elements_list[i])):
                self.elements.append(elements_list[i][j])

        return self.elements

    def run_with_thread_pool_executor(self):
        """
        Run Tesseract OCR on the given images, on multiple threads using concurrent.futures.ThreadPoolExecutor
        :returns: a list of DocumentElement
        """
        self.startTime = time.time()
        self.times = []
        self.timesFromStart = []
        times_sum = []
        # print('start time ocr:', int(self.starTime))

        self.elements = []
        page_count = len(self.images)
        elements_of_pages = []

        for i in range(page_count):
            elements_of_pages.append([])
            self.times.append(0)
            times_sum.append(0)
            self.timesFromStart.append(0)

        self.__futures = []
        # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        self.__executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        # futures = []
        for i in range(page_count):
            a_result = self.__executor.submit(self.__task_for_thread_pool, elements_of_pages[i], i)
            self.__futures.append(a_result)

        for future in as_completed(self.__futures):
            result = future.result()
            for j in range(1, len(result)):
                elements_of_pages[result[0]].append(result[j])

        for i in range(len(elements_of_pages)):
            for j in range(len(elements_of_pages[i])):
                self.elements.append(elements_of_pages[i][j])

        # file_ocr = open('ocr_time.txt', 'w')
        # self.timesFromStart.sort()

        for i in range(len(self.times)):
            if i == 0:
                times_sum[i] = self.times[i]
            else:
                times_sum[i] = round(times_sum[i-1] + self.times[i], 4)

            # file_ocr.write(str(timesSum[i]) + ' ')
        # file_ocr.write('\n')

        # for i in range(len(self.timesFromStart)):
        #    file_ocr.write(str(self.timesFromStart[i])+' ')

        # file_ocr.close()
        # print('times:', self.times)
        # print('timesSum: ', timesSum)
        # print('timesFromStart: ', self.timesFromStart)
        self.__executor.shutdown()

        return self.elements

    def shutdown_thread_pool_executor(self):
        """
        Shutdown the ThreadPoolExecutor if still running
        """
        self.__is_shutDown = True

        if self.__executor is not None:
            print('shutdown executor')
            self.__executor.shutdown(wait=False, cancel_futures=True)

    def __task_for_thread_pool(self, elements, num):
        """
        Funciton for ThreadPoolExecutor
        """
        start_time = time.time()
        elements2 = [num]
        if num < 0 or num >= len(self.images):
            return

        data = pytesseract.image_to_data(self.images[num], 
                                         lang='hun', 
                                         output_type=pytesseract.Output.DICT)

        """ for j in range(len(data['level'])):
            elements2.append(
                DocumentElement(data['level'][j], 
                                data['left'][j], 
                                data['top'][j], 
                                data['width'][j], 
                                data['height'][j], 
                                data['text'][j])
            ) """
        elements2 += [DocumentElement(data['level'][j], 
                                      data['left'][j], 
                                      data['top'][j], 
                                      data['width'][j], 
                                      data['height'][j], 
                                      data['text'][j]) 
                      for j in range(len(data['level']))]
        self.lock.acquire()

        try:
            self.progress += 1
            end_time = time.time()
            # print('end time:', int(end_time))
            delta_time = end_time - start_time
            # print(str(num) +' : ' +  str(int(deltaTime)), end=' ')
            round_time = round(delta_time, 4)
            self.times[num] = round_time
            delta_time_from_start = end_time-self.startTime
            self.timesFromStart[num] = round(delta_time_from_start, 4)
        finally:
            self.lock.release()

        if self.parent is not None and not self.__is_shutDown:
            self.lock2.acquire()
            try:
                self.parent.page_processed(self.get_count(), self.progress)
            finally:
                self.lock2.release()

        return elements2

    def __task_for_threads(self, elements, fr, to):
        # elements = []
        if fr < 0:
            return
        if to > len(self.images):
            return

        for i in range(fr, to):
            data = pytesseract.image_to_data(self.images[i], 
                                             lang='hun', 
                                             output_type=pytesseract.Output.DICT)

            for j in range(len(data['level'])):
                self.elements.append(
                    DocumentElement(data['level'][j],
                                    data['left'][j],
                                    data['top'][j],
                                    data['width'][j],
                                    data['height'][j],
                                    data['text'][j])
                )
            self.lock.acquire()

            try:
                self.progress += 1
            finally:
                self.lock.release()

            if self.parent is not None:
                self.lock2.acquire()
                try:
                    self.parent.page_processed(self.get_count(), self.progress)
                finally:
                    self.lock2.release()


class PyMuPdfTextExtractModule:
    """
    Class for extracting original OCR and layout information from the given pdf
    """
    def __init__(self, parent=None):
        self.elements = []
        self.parent = parent
        self.progress = 0
        self.pageCount = 0
        self.pdf = None
        self.path = None

    def set_path(self, path):
        """
        Set the path of pdf file
        :param path: str, path of pdf
        """
        self.path = path

    def set_pdf(self, pdf):
        self.pdf = pdf

    def get_count(self):
        """
        :returns: int, number of pages in the pdf
        """
        return self.pageCount

    def get_progress(self):
        """
        :returns: int, how many pages are processed
        """
        return self.progress

    def run(self):
        """
        Extracts Ocr and layout information from the pdf.
        :returns: list[DocumentElement]
        """
        self.elements = []
        pdf_file = fitz.open(self.path)
        self.pageCount = len(pdf_file)
        for pageIndex in range(len(pdf_file)):
            page = pdf_file[pageIndex]
            data = page.get_text('dict')

            self.elements.append(
                DocumentElement(1,
                                0,
                                0,
                                data['width'],
                                data['height'])
            )
            for block in data['blocks']:

                if block['type'] == 0:
                    self.elements.append(
                        DocumentElement(2, 
                                        block['bbox'][0],
                                        block['bbox'][1],
                                        block['bbox'][2]-block['bbox'][0],
                                        block['bbox'][3]-block['bbox'][1])
                        )

                    self.elements.append(
                        DocumentElement(3,
                                        block['bbox'][0],
                                        block['bbox'][1],
                                        block['bbox'][2]-block['bbox'][0],
                                        block['bbox'][3]-block['bbox'][1])
                        )
                    for line in block['lines']:
                        self.elements.append(
                            DocumentElement(4,
                                            line['bbox'][0],
                                            line['bbox'][1],
                                            line['bbox'][2]-line['bbox'][0],
                                            line['bbox'][3]-line['bbox'][1])
                            )
                        for span in line['spans']:
                            self.elements.append(
                                DocumentElement(5,
                                                span['bbox'][0],
                                                span['bbox'][1],
                                                span['bbox'][2]-span['bbox'][0],
                                                span['bbox'][3]-span['bbox'][1],
                                                span['text'])
                                )

        return self.elements

    def get_page(self, page_number: int):
        """
        returns the DocumentElement list of the page based on the given page number
        :param page_number: int
        """
        elements = []
        pdf_file = fitz.open(self.path)
        page_count = len(pdf_file)

        if page_number > page_count or page_number < 0:
            print('Invalid page number')

        page = pdf_file[page_number]
        data = page.get_text('dict')

        elements.append(
            DocumentElement(1,
                            0,
                            0,
                            data['width'],
                            data['height'])
        )
        for block in data['blocks']:

            if block['type'] == 0:
                elements.append(
                    DocumentElement(2,
                                    block['bbox'][0],
                                    block['bbox'][1],
                                    block['bbox'][2]-block['bbox'][0],
                                    block['bbox'][3]-block['bbox'][1])
                    )

                elements.append(
                    DocumentElement(3,
                                    block['bbox'][0],
                                    block['bbox'][1],
                                    block['bbox'][2]-block['bbox'][0],
                                    block['bbox'][3]-block['bbox'][1])
                    )
                for line in block['lines']:
                    elements.append(
                        DocumentElement(4,
                                        line['bbox'][0],
                                        line['bbox'][1],
                                        line['bbox'][2]-line['bbox'][0],
                                        line['bbox'][3]-line['bbox'][1])
                        )
                    for span in line['spans']:
                        elements.append(
                            DocumentElement(5,
                                            span['bbox'][0],
                                            span['bbox'][1],
                                            span['bbox'][2]-span['bbox'][0],
                                            span['bbox'][3]-span['bbox'][1],
                                            span['text'])
                            )
        return elements

    def get_page_plain_text(self, page_number: int):
        """
        returns the DocumentElement list of the page based on the given page number
        :param page_number: int
        """
        pdf_file = fitz.open(self.path)
        page = pdf_file[page_number]
        return page.get_text()


class FastLayoutTypeAnalyzeModule:
    """
    Class for analyzing the type of the layout elements
    """
    def __init__(self, parent=None):
        self.images = None
        self.elements = []
        self.parent = parent
    
    def set_elements(self, elements):
        self.elements = elements
    
    def set_images(self, images):
        self.images = images

    def run(self):
        avg_font_size = self.get_average_font_size()
        
        #for i in range(len(self.elements)):
        for i, elem in enumerate(self.elements):
            if elem.level == 2:
                text_blocks = 0
                text_lines = 0
                strings = 0
                for j in range(i+1, len(self.elements)):
                    if self.elements[j].level == 2:
                        break
                    elif self.elements[j].level == 3:
                        text_blocks += 1
                    elif self.elements[j].level == 4:
                        text_lines += 1
                    elif self.elements[j].level == 5:
                        strings += 1
                
                if text_blocks == 1 and text_lines == 1 and strings == 1 and self.elements[i+3].text.replace(' ', '').replace('.', '').isnumeric():
                    # print(self.elements[i+3].text)
                    self.elements[i].type = 'page_number'
                elif text_lines <= 3 and strings <= 10:
                    tmp_str = ""
                    for j in range(i+1, len(self.elements)):
                        if self.elements[j].level == 2:
                            break
                        elif self.elements[j].level == 5:
                            tmp_str = tmp_str + ' ' + self.elements[j].text
                    
                    if tmp_str.isupper():
                        self.elements[i].type = 'title'
                else:
                    self.elements[i].type = 'paragraph'

            
        return self.elements

    def get_average_font_size(self):
        numbers = []
        for elem in self.elements:
            if elem.level == 5:
                numbers.append(elem.height)

        return sum(numbers)/len(numbers) if len(numbers)>0 else 0


class DocumentElementstToAlto:
    """
    Class wich converts a list of DocumentElement-s to a formatted alto xml and saves it to a file on the given path.
    """
    def __init__(self, parent=None):
        self.outputPath = None
        self.parent = parent
        self.elements = None

    def set_output_path(self, output_path):
        """
        Sets the output file path
        :param output_path: str, path of the output file
        """
        self.outputPath = output_path

    def set_elements(self, elements):
        """
        Sets the list of DocumentElements
        :param elements: list[DocumentElement]
        """
        self.elements = elements

    def run(self, with_header=True):
        """
        Creates an alto xml based on the list of DocumentElements, and saves it to the given path
        """
        current_level = 1
        alto_types = ['Page', 'PrintSpace', 'ComposedBlock', 'TextBlock', 'TextLine', 'String']
        page_id = 1

        out_file = open(self.outputPath, 'w', encoding='utf-8')
        if with_header:
            out_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            out_file.write('''<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v3# http://www.loc.gov/alto/v3/alto-3-0.xsd">\n''')

            # out_file.write('<Description></Description>\n')
            out_file.write('<Layout>\n')

        for i in range(len(self.elements)):
            elem = self.elements[i]
            if elem.level > current_level:
                current_level = elem.level
            elif elem.level < current_level:
                while current_level >= elem.level:
                    if current_level != 5:
                        for k in range(current_level):
                            out_file.write('    ')
                        out_file.write('</' + alto_types[current_level] + '>\n')
                        if current_level == 1:
                            out_file.write('    </Page>\n')
                    current_level -= 1
                current_level = elem.level
            elif elem.level == current_level and current_level != 5 and i != 0:
                if current_level == 2:
                    out_file.write('        </ComposedBlock>\n')
                else:
                    out_file.write('    </PrintSpace>\n')
                    out_file.write('    </Page>\n')

            if current_level == 1:
                out_file.write('    <Page ID="page' + str(page_id) + '" PHYSICAL_IMG_NR="' + str(page_id) + '">\n')
                page_id += 1

            for k in range(current_level):
                out_file.write('    ')

            string = elem.convert_to_alto_tag(('id' + str(i)))
            out_file.write(string)
            out_file.write('\n')

        while current_level >= 0:
            if current_level != 5:
                for k in range(current_level):
                    out_file.write('    ')
                out_file.write('</' + alto_types[current_level] + '>\n')
            current_level -= 1

        if with_header:
            out_file.write('</Layout></alto>')
        out_file.close()

    def get_string(self, elements, page_id=0):
        current_level = 1
        alto_types = ['Page', 'PrintSpace', 'ComposedBlock', 'TextBlock', 'TextLine', 'String']
        out_str = ''

        for i in range(len(self.elements)):
            elem = self.elements[i]
            if elem.level > current_level:
                current_level = elem.level
            elif elem.level < current_level:
                while current_level >= elem.level:
                    if current_level != 5:
                        for k in range(current_level):
                            out_str += '    '
                        out_str += ('</' + alto_types[current_level] + '>\n')
                        if current_level == 1:
                            out_str += '    </Page>\n'
                    current_level -= 1
                current_level = elem.level
            elif elem.level == current_level and current_level != 5 and i != 0:
                if current_level == 2:
                    out_str += '        </ComposedBlock>\n'
                else:
                    out_str += '    </PrintSpace>\n'
                    out_str += '    </Page>\n'

            if current_level == 1:
                out_str += ('    <Page ID="page' + str(page_id) + '" PHYSICAL_IMG_NR="' + str(page_id) + '">\n')
                # page_id += 1

            for k in range(current_level):
                out_str += '    '

            string = elem.convert_to_alto_tag(('id' + str(i)))
            out_str += string
            out_str += '\n'

        while current_level >= 0:
            if current_level != 5:
                for k in range(current_level):
                    out_str += '    '
                out_str += ('</' + alto_types[current_level] + '>\n')
            current_level -= 1

        return out_str


class DocumentElementsToText:
    def __init__(self, parent=None):
        self.outputPath = None
        self.parent = parent
        self.elements = None

    def set_elements(self, elements):
        """
        Sets the list of DocumentElements
        :param elements: list[DocumentElement]
        """
        self.elements = elements
    
    def set_output_path(self, output_path):
        """
        Sets the output file path
        :param output_path: str, path of the output file
        """
        self.outputPath = output_path

    def run(self):
        out_file = open(self.outputPath, 'w', encoding='utf-8')

        for i in range(len(self.elements)):
            elem = self.elements[i]

            if elem.level == 2:
                out_file.write('\n')
            if elem.level == 4:
                out_file.write('\n')
            if elem.level == 5:
                out_file.write(elem.text + ' ')

        out_file.close()
    
    def get_string(self):
        result = ''
        for i in range(len(self.elements)):
            elem = self.elements[i]

            is_new = False
            if elem.level == 2:
                result += '\n' + '' if elem.type == '' else '\n\n<' + elem.type + '>'
            if elem.level == 4:
                result += '\n'
            if elem.level == 5:
                result += elem.text + ' '

        return result


class DocumentElementsToPdf:
    def __init__(self, parent=None):
        self.outputPath = None
        self.parent = parent
        self.elements = []
        self.images = None

    def set_output_path(self, output_path):
        """
        Sets the output file path
        :param output_path: str, path of the output file
        """
        self.outputPath = output_path

    def set_elements(self, elements):
        """
        Sets the list of DocumentElements
        :param elements: list[DocumentElement]
        """
        self.elements = elements

    def set_images(self, images: list):
        """
        Set the images list of images
        :param images: list of PIL images
        """
        self.images = images

    def run(self):
        out_document = fitz.open()
        pages = []
        page_index = 0

        originalw = self.elements[0].width
        originalh = self.elements[0].height

        multw = 555/originalw
        multh = 842/originalh

        for i, element in enumerate(self.elements):
            if element.level == 1:
                pages.append(out_document.new_page())
                page_index += 1
                pages[page_index-1].set_mediabox([0, 0, 555, 842])
                # page.set_mediabox([0, 0, 1618, 2429])
            elif element.level == 5:
                # multl = 555/1618
                # multt = 842/2429
                point = fitz.Point(element.left*multw, element.top*multh+11)
                fontfile = 'c:/windows/fonts/Calibri.ttf'
                pages[page_index-1].insert_text(point,
                                                str(element.text),
                                                fontsize=13,
                                                encoding=fitz.TEXT_ENCODING_LATIN,
                                                fontfile=fontfile,
                                                fontname='Calibri')
        # print('output', self.outputPath)

        # for i in range(len(self.images)):
            # pages[i].insert_image(self.images[i])
        out_document.save(self.outputPath)
