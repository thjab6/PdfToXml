import threading
import tkinter as ttk
from tkinter import Variable, filedialog
from tkinter.ttk import Progressbar, Combobox
from PdfToXml import *
import time
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import os


TRANSFORM_PDF_TO_IMAGES = 'Pdf képekké alakítása'
TRANSFORM_PDF_DONE = 'Átalakítás kész'
TESSERACT_OCR_PROCESSING = 'Tesseract Ocr feldolgozás'
PYMUPDF_PROCESSING = 'PyMuPdf feldolgozás'
SAVING_FILE = 'Fájl mentése'
ALL_DONE = 'Kész'
FILE_OPEN_BUTTON = 'Fájl megnyitása'
FILE_SAVE_BUTTON = 'Mentés helye'
START_BUTTON = 'Indítás'
LOAD_MODES_LABEL = 'Betöltési módok:'
EXPORT_IMAGES_RADIO = 'Képek exportálása'
RENDER_PAGES_RADIO = 'oldalak leképezése'
OCR_MODES_LABEL = 'Ocr módok:'
MUPDF_RADIO = 'MuPdf'
TESSERACT_OCR_RADIO = 'Többszálú Tesseract-ocr'
TESSERACT_SINGLE_RADIO = 'Egyszálú Tesseract-ocr'
SAVE_MODES_LABEL = 'Mentési módok'
SAVE_AS_ALTO_RADIO = 'alto xml'
SAVE_AS_PDF_RADIO = 'pdf'
PDF_PREVIEW_LABEL = 'Pdf előnézet'
MUPDF_PREVIEW_LABEL = 'MuPdf előnézet'
TESSERRACT_OCR_PREVIEW_LABEL = 'Tesseract ocr előnézet'
OPEN_FILE_DIALOG = 'Fájl megnyitása'
SAVE_FILE_DIALOG = 'Fájl mentése'
LOADING = 'betöltés'

HELP_WINDOW_NAME = 'Súgó'
help_file = open("manual.txt", "r", encoding='utf-8')
HELP_TEXT = help_file.read()
help_file.close()


class ModelForView:
    """
    Model that contains a version of PpfToXml pipeline
    It can convert a pdf to alto xml.
    """

    def __init__(self):
        self.controller = None
        self.processInfo = ""
        self.convert_pdf_to_images = FastImgLoad()
        self.pymuStr = ""
        self.tesseractStr = ""
        self.images = None
        self.input_path = None
        self.output_path = None
        self.__proc1 = None
        self.__data = None

    def set_input_path(self, input_path):
        """
        set the input pdf
        :param input_path:
        :return:
        """
        self.input_path = input_path
        self.load_pdf_images()

    def set_output_path(self, output_path):
        """
        set the output xml
        :param output_path:
        :return:
        """
        self.output_path = output_path

    def set_convert_pdf_to_images(self, convert_pdf_to_images):
        """
        set the pdf to image converter class
        """
        self.convert_pdf_to_images = convert_pdf_to_images
        self.load_pdf_images()

    def load_pdf_images(self):
        """
        loads the pdf from the previously given path
        """
        self.__set_process_info(TRANSFORM_PDF_TO_IMAGES)
        if self.input_path is not None:
            self.images = self.convert_pdf_to_images.convert(self.input_path)
        self.__set_process_info('átalakítás kész')

    def run(self, process_type, load_mode, save_type, layoutRecognize, poemRecognize):
        """
        run the pipeline
        :return:
        """
        print('recognize:', layoutRecognize, poemRecognize)
        star_time = time.time()
        # print('start time:', star_time)

        if self.input_path is None or not os.path.exists(self.input_path):
            self.__send_message(title='hiba',
                                message='A bemeneti útvonal nem létezik.')
            return

        # print('output:', self.output_path)
        if self.output_path is None:
            self.__send_message(title='hiba',
                                message='A kimeneti útvonal nem létezik.')
            return

        output_list = self.output_path.split('/')
        output_list.pop()
        if not os.path.exists(''.join(list(map(lambda s: s + '/', output_list)))):
            self.__send_message(title='hiba',
                                message='A kimeneti útvonal nem helyes.')
            return

        self.controller.set_running(True)

        # convert pdf to images
        self.__set_process_info(TRANSFORM_PDF_TO_IMAGES)
        if load_mode == 'EXPORT':
            self.convert_pdf_to_images = PyMuPdfModule(parent=self)
        elif load_mode == 'RENDER':
            self.convert_pdf_to_images = PdfToImgModule(parent=self)
        #self.convert_pdf_to_images.convert(self.input_path)

        end_time = time.time()
        # print("end time:", int(end_time))
        delta_time = end_time - star_time
        # print("delta time", int(delta_time))

        # process images

        if process_type == 'TESSERACT':
            self.__set_process_info(TESSERACT_OCR_PROCESSING)
            self.__proc1 = TesseractModule(self)
            self.__proc1.set_images(self.images)
            # self.__data = self.__proc1.run_with_threading()
            self.__data = self.__proc1.run_with_thread_pool_executor()
            # self.__data = self.__proc1.run()

        elif process_type == 'PYMUPDF':
            self.__set_process_info(PYMUPDF_PROCESSING)
            __proc1 = PyMuPdfTextExtractModule(self)
            __proc1.set_path(self.input_path)
            self.__data = __proc1.run()
        elif process_type == 'TESSERACT_S':
            self.__set_process_info(TESSERACT_OCR_PROCESSING)
            self.__proc1 = TesseractModule(self)
            self.__proc1.set_images(self.images)
            self.__data = self.__proc1.run()
        elif process_type == 'Mu+Tesseract':
            self.__set_process_info("PyMuPdf - layout exportálás")
            __proc1 = PyMuPdfTextExtractModule(self)
            __proc1.set_path(self.input_path)
            self.__data = __proc1.run()

            self.__set_process_info("Tesseract ocr - szöveg felismerés")
            proc2 = TesseractTextOnly(self)
            proc2.set_elements(self.__data)
            proc2.set_images(self.images)

            self.__data = proc2.run()

        # layout type analyze
        # self.__set_process_info("Layout típusok felismerése")
        # self.__proc2 = FastLayoutTypeAnalyzeModule(self)
        # self.__proc2.set_elements(elements=self.__data)
        #self.__data = self.__proc2.run()

        
        #layout clustering
        if layoutRecognize == 'Recognize':
            self.__set_process_info('Layout klaszterezése')
            self.__proc3 = LayoutClustering(self)
            self.__proc3.set_elements(elements=self.__data)
            self.__data = self.__proc3.run('KMEANS')

        # block axis
        self.__set_process_info("Blokk tengelyek megállapítása")
        self.__proc21 = BlockAxis(self)
        self.__proc21.set_elements(elements=self.__data)
        self.__data = self.__proc21.run()
        #print(self.__data)

        if poemRecognize == 'Recognize':
            # block reordering
            self.__set_process_info("Blokkok újra rendezése")
            self.__proc20 = BlockReordering(self)
            self.__proc20.set_elements(elements=self.__data)
            self.__data = self.__proc20.run()

            #poem
            self.__set_process_info('vers')
            self.__proc4 = PoemAnalyzer(self)
            self.__proc4.set_elements(elements=self.__data)
            self.__proc4.run()

        # save file
        self.__set_process_info(SAVING_FILE)
        self.save(save_type)

        end_time = time.time()
        # print("end time:", int(end_time))
        delta_time = end_time - star_time
        print('run time', round(delta_time, 3), 'sec')

        self.controller.set_running(False)
        self.__set_process_info(ALL_DONE)
        self.__send_message(ALL_DONE)

        if False and process_type == 'TESSERACT':
            plt.grid()

            plt.plot(list(range(1, len(self.images) + 1)),
                     self.__proc1.timesFromStart)
            plt.xticks(list(range(1, len(self.images) + 1)))
            plt.show()

    def get_preview(self, page_number: int, preview1_type, preview2_type):
        """
        Gets xml preview of the given page
        :param page_number: int
        :param preview1_type: 'MUPDF_XML', 'TESSERACT_XML', 'MUPDF_TEXT', 'TESSERACT_TEXT'
        """
        converter = DocumentElementstToAlto()

        self.pymuStr = ''
        self.tesseractStr= ''

        if preview1_type == 'MUPDF_XML':
            pymu = PyMuPdfTextExtractModule()
            pymu.set_path(self.input_path)
            pymu_data = pymu.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(pymu_data)
            pymu_data = layout.run()

            converter.set_elements(pymu_data)
            self.pymuStr = converter.get_string(pymu_data)
        elif preview1_type == 'TESSERACT_XML':
            tesseract = TesseractModule()
            tesseract.set_images(self.images)
            tesseract_data = tesseract.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(tesseract_data)
            tesseract_data = layout.run()

            converter.set_elements(tesseract_data)
            self.pymuStr = converter.get_string(tesseract_data)
        elif preview1_type == 'MUPDF_TEXT':
            pymu = PyMuPdfTextExtractModule()
            pymu.set_path(self.input_path)
            self.pymuStr = pymu.get_page_plain_text(page_number)
        elif preview1_type == 'TESSERACT_TEXT':
            tesseract = TesseractModule()
            tesseract.set_images(self.images)
            self.pymuStr = tesseract.get_page_plain_text(page_number)
        elif preview1_type == 'MUPDF_TEXT+TYPES':
            pymu = PyMuPdfTextExtractModule()
            pymu.set_path(self.input_path)
            elements = pymu.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(elements)
            elements = layout.run()

            converter2 = DocumentElementsToText()
            converter2.set_elements(elements)
            self.pymuStr = converter2.get_string()
        elif preview1_type == 'TESSERACT_TEXT+TYPEST':
            tesseract = TesseractModule()
            tesseract.set_images(self.images)
            elements = tesseract.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(elements)
            elements = layout.run()

            converter2 = DocumentElementsToText()
            converter2.set_elements(elements)
            self.pymuStr = converter2.get_string()
        else:
            self.pymuStr = 'hiba: ismeretlen előnézeti típus'

        # preview2
        if preview2_type == 'MUPDF_XML':
            pymu = PyMuPdfTextExtractModule()
            pymu.set_path(self.input_path)
            pymu_data = pymu.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(pymu_data)
            pymu_data = layout.run()

            converter.set_elements(pymu_data)
            self.tesseractStr = converter.get_string(pymu_data)
        elif preview2_type == 'TESSERACT_XML':
            tesseract = TesseractModule()
            tesseract.set_images(self.images)
            tesseract_data = tesseract.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(tesseract_data)
            tesseract_data = layout.run()

            converter.set_elements(tesseract_data)
            self.tesseractStr = converter.get_string(tesseract_data)
        elif preview2_type == 'MUPDF_TEXT':
            pymu = PyMuPdfTextExtractModule()
            pymu.set_path(self.input_path)
            self.tesseractStr = pymu.get_page_plain_text(page_number)
        elif preview2_type == 'TESSERACT_TEXT':
            tesseract = TesseractModule()
            tesseract.set_images(self.images)
            self.tesseractStr = tesseract.get_page_plain_text(page_number)
        elif preview2_type == 'MUPDF_TEXT+TYPES':
            pymu = PyMuPdfTextExtractModule()
            pymu.set_path(self.input_path)
            elements = pymu.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(elements)
            elements = layout.run()

            converter2 = DocumentElementsToText()
            converter2.set_elements(elements)
            self.tesseractStr = converter2.get_string()
        elif preview2_type == 'TESSERACT_TEXT+TYPEST':
            tesseract = TesseractModule()
            tesseract.set_images(self.images)
            elements = tesseract.get_page(page_number)

            layout = FastLayoutTypeAnalyzeModule()
            layout.set_elements(elements)
            elements = layout.run()

            converter2 = DocumentElementsToText()
            converter2.set_elements(elements)
            self.tesseractStr = converter2.get_string()
        else:
            self.tesseractStr = 'hiba: ismeretlen előnézeti típus'
        
        self.controller.set_xml_preview(self.pymuStr, self.tesseractStr)

    def save(self, save_type: str = 'ALTO'):
        """
        Save the xml to the previously given url
        :param save_type: str "ALTO" or "PDF"
        """
        if save_type == 'PDF':
            converter = DocumentElementsToPdf()
            converter.set_elements(self.__data)
            converter.set_images(self.images)
            converter.set_output_path(self.output_path)
            converter.run()
        elif save_type == 'TXT':
            converter = DocumentElementsToText()
            converter.set_elements(self.__data)
            converter.set_output_path(self.output_path)
            converter.run()
        else:
            converter = DocumentElementstToAlto()
            converter.set_elements(self.__data)
            converter.set_output_path(self.output_path)
            converter.run()

    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller

    def page_processed(self, page_count, current_page):
        if self.controller is not None:
            self.controller.page_processed(page_count, current_page)

    def __set_process_info(self, process_info):
        if self.controller is not None:
            self.controller.set_process_info(process_info)

    def __send_message(self, message, title=None):
        if self.controller is not None:
            self.controller.send_message(message, title)

    def stop_all_process(self):
        try:
            self.__proc1.shutdown_thread_pool_executor()
        except Exception as e:
            print(e)

    def get_images(self):
        return self.images

class View(ttk.Frame):
    """
    View class for PdfToXml application
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.controller = None

        # Frames
        self.ioFrame = ttk.Frame(self)
        self.settingsFrame = ttk.Frame(self)
        self.pdfFrame = ttk.Frame(self)
        self.pymuxmlFrame = ttk.Frame(self, padx=2)
        self.tesseractxmlFrame = ttk.Frame(self, padx=2)

        self.ioFrame.grid(row=0, column=0)
        self.settingsFrame.grid(row=0, column=1)
        self.pdfFrame.grid(row=1, column=0)
        self.pymuxmlFrame.grid(row=1, column=1)
        self.tesseractxmlFrame.grid(row=1, column=2)

        self.manual = ttk.Button(self, text='Súgó', command=self.__click_help)
        self.manual.grid(row=0, column=2, sticky='ne')

        self.inputEntry = ttk.Entry(self.ioFrame, width=50)
        self.inputEntry.grid(row=0, column=0)

        # input button
        self.inputButton = ttk.Button(self.ioFrame,
                                      text=FILE_OPEN_BUTTON,
                                      padx=20,
                                      pady=10,
                                      width=10,
                                      command=self.__click_open_file)
        self.inputButton.grid(row=0, column=1)

        # output field
        self.outputEntry = ttk.Entry(self.ioFrame, width=50)
        self.outputEntry.grid(row=1, column=0)

        # output button
        self.outputButton = ttk.Button(self.ioFrame,
                                       text=FILE_SAVE_BUTTON,
                                       padx=20,
                                       pady=10,
                                       width=10,
                                       command=self.__click_save_file)
        self.outputButton.grid(row=1, column=1)

        # start button
        self.startButton = ttk.Button(self.ioFrame,
                                      text=START_BUTTON,
                                      padx=20,
                                      pady=10,
                                      height=4,
                                      command=self.__start_processing_command)
        self.startButton.grid(row=0, column=2, rowspan=2)

        # create label
        self.processInfo = ttk.Label(self.ioFrame, text="")
        self.processInfo.grid(row=2, column=0)

        self.progBar = Progressbar(self.ioFrame, orient=ttk.HORIZONTAL, length=500, mode='determinate')
        self.progBar.grid(row=3, column=0, columnspan=3)

        # load modes
        self.loadModeVar = ttk.StringVar()  # 'EXPORT', 'RENDER'
        self.loadModeVar.set('EXPORT')

        self.settingsFrame.loadModes = ttk.Frame(self.settingsFrame)
        self.settingsFrame.loadModes.grid(row=0, column=0, sticky='w')

        self.settingsFrame.loadModes.label = ttk.Label(self.settingsFrame.loadModes, text=LOAD_MODES_LABEL)
        self.settingsFrame.loadModes.imgExportRadio = ttk.Radiobutton(self.settingsFrame.loadModes,
                                                                      text=EXPORT_IMAGES_RADIO,
                                                                      variable=self.loadModeVar,
                                                                      value='EXPORT')
        self.settingsFrame.loadModes.renderImagesRadio = ttk.Radiobutton(self.settingsFrame.loadModes,
                                                                         text=RENDER_PAGES_RADIO,
                                                                         variable=self.loadModeVar,
                                                                         value='RENDER')
        self.settingsFrame.loadModes.label.grid(row=0, column=0, sticky='w')
        self.settingsFrame.loadModes.imgExportRadio.grid(row=1, column=0)
        self.settingsFrame.loadModes.renderImagesRadio.grid(row=1, column=1)

        # ocr modes
        self.processType = ttk.StringVar()  # 'PYMUPDF', 'TESSERACT'
        self.processType.set('PYMUPDF')

        self.settingsFrame.ocrModes = ttk.Frame(self.settingsFrame)
        self.settingsFrame.ocrModes.grid(row=1, column=0, sticky='w')

        self.optionProcessLabel = ttk.Label(self.settingsFrame.ocrModes, text=OCR_MODES_LABEL)
        self.PyMuProcessRadio = ttk.Radiobutton(self.settingsFrame.ocrModes,
                                                text=MUPDF_RADIO,
                                                variable=self.processType,
                                                value='PYMUPDF')
        self.TesseractRadio = ttk.Radiobutton(self.settingsFrame.ocrModes,
                                              text=TESSERACT_OCR_RADIO,
                                              variable=self.processType,
                                              value='TESSERACT')
        self.TesseractSingleRadio = ttk.Radiobutton(self.settingsFrame.ocrModes,
                                                    text=TESSERACT_SINGLE_RADIO,
                                                    variable=self.processType,
                                                    value='TESSERACT_S')
        self.MuAndTesseractRadio = ttk.Radiobutton(self.settingsFrame.ocrModes,
                                                   text='MuPdf+Tesseract',
                                                   variable=self.processType,
                                                   value='Mu+Tesseract')

        self.optionProcessLabel.grid(row=0, column=0)
        self.PyMuProcessRadio.grid(row=1, column=0)
        self.TesseractRadio.grid(row=1, column=1)
        self.TesseractSingleRadio.grid(row=1, column=2)
        # self.MuAndTesseractRadio.grid(row=1, column=2)

        # save modes
        self.saveType = ttk.StringVar()  # 'ALTO', 'PDF'
        self.saveType.set('ALTO')

        self.settingsFrame.saveModes = ttk.Frame(self.settingsFrame)
        self.settingsFrame.saveModes.grid(row=0, column=1, sticky='w')

        self.saveModesLabel = ttk.Label(self.settingsFrame.saveModes, text=SAVE_MODES_LABEL)
        self.saveAsAltoRadio = ttk.Radiobutton(self.settingsFrame.saveModes,
                                               text=SAVE_AS_ALTO_RADIO,
                                               variable=self.saveType,
                                               value='ALTO')
        self.saveAsPdfRadio = ttk.Radiobutton(self.settingsFrame.saveModes,
                                              text=SAVE_AS_PDF_RADIO,
                                              variable=self.saveType,
                                              value='PDF')
        self.saveAsTxtRadio = ttk.Radiobutton(self.settingsFrame.saveModes,
                                              text='txt',
                                              variable=self.saveType,
                                              value='TXT')

        self.saveModesLabel.grid(row=0, column=0)
        self.saveAsAltoRadio.grid(row=1, column=0, sticky='w')
        # self.saveAsPdfRadio.grid(row=1, column=1)
        self.saveAsTxtRadio.grid(row=1, column=2)

        # layout modes
        self.settingsFrame.layoutModes = ttk.Frame(self.settingsFrame)
        self.settingsFrame.layoutModes.grid(row=1, column=1, sticky='w')

        self.layoutModesLabel = ttk.Label(self.settingsFrame.layoutModes, text='Layout módok')

        self.layoutRecognize = ttk.StringVar() 
        self.layoutRecognize.set('Recognize')
        self.layoutRecognizeCheck = ttk.Checkbutton(self.settingsFrame.layoutModes, 
                                                    text='Layout típusok',
                                                    variable=self.layoutRecognize,
                                                    onvalue='Recognize',
                                                    offvalue='None')

        self.poemRecognize = ttk.StringVar()
        self.poemRecognize.set('None')
        self.poemRecognizeCheck = ttk.Checkbutton(self.settingsFrame.layoutModes, 
                                                    text='Versek',
                                                    variable=self.poemRecognize,
                                                    onvalue='Recognize',
                                                    offvalue='None')


        self.layoutModesLabel.grid(row=0, column=0, sticky='w')
        self.layoutRecognizeCheck.grid(row=1, column=0)
        self.poemRecognizeCheck.grid(row=1, column=1)


        # pdf view
        self.pdfPages = []

        self.pdfViewLabel = ttk.Label(self.pdfFrame, text=PDF_PREVIEW_LABEL)
        self.pdfViewLabel.grid(row=0, column=0, columnspan=3)

        self.pdfView = ttk.Label(self.pdfFrame)
        self.__placeHolderImage = Image.new(mode='RGB', size=(400, 600))
        self.__placeHolderImageTk = ImageTk.PhotoImage(self.__placeHolderImage)
        self.pdfView.config(image=self.__placeHolderImageTk)
        self.pdfView.grid(row=1, column=0, columnspan=3)

        self.pageIndVar = ttk.StringVar(value=str(0))

        self.buttonBack = ttk.Button(self.pdfFrame, text="<-", command=self.__click_back_button_command)
        self.buttonBack.grid(row=2, column=0)
        self.pdfSpinBox = ttk.Spinbox(self.pdfFrame, from_=0, to=1000, textvariable=self.pageIndVar)
        self.pdfSpinBox.grid(row=2, column=1)
        self.buttonForward = ttk.Button(self.pdfFrame, text="->", command=self.__click_forward_button_command)
        self.buttonForward.grid(row=2, column=2)

        # preview1
        self.previewTypes = {
            'MuPdf xml' : 'MUPDF_XML',
            'Tesseract xml' : 'TESSERACT_XML',
            'MuPdf szöveg' : 'MUPDF_TEXT',
            'Tesseract szöveg' : 'TESSERACT_TEXT',
            'MuPdf szöveg+típusok' : 'MUPDF_TEXT+TYPES',
            'Tesseract szöveg+típusok' : 'TESSERACT_TEXT+TYPEST'
        }

        # self.preview1Label = ttk.Label(self.pymuxmlFrame, text=MUPDF_PREVIEW_LABEL)
        # self.preview1Label.pack()

        self.preview1Var = ttk.StringVar()
        self.preview1Var.set('MuPdf szöveg')
        self.preview1Var.trace_add('write', lambda a,b,c : 
                                    self.controller.get_preview(self.pageInd, 
                                                                self.previewTypes[self.preview1Var.get()],
                                                                self.previewTypes[self.preview2Var.get()]))
        self.preview1Combobox = Combobox(self.pymuxmlFrame,textvariable=self.preview1Var)
        self.preview1Combobox['values'] = list(self.previewTypes.keys())
        self.preview1Combobox.pack()

        self.preview1 = ttk.Text(self.pymuxmlFrame, width=80, height=37, wrap=ttk.NONE)
        self.preview1ScrollX = ttk.Scrollbar(self.pymuxmlFrame, orient='horizontal', command=self.preview1.xview)
        self.preview1['xscrollcommand'] = self.preview1ScrollX.set
        self.preview1ScrollY = ttk.Scrollbar(self.pymuxmlFrame, orient='vertical', command=self.preview1.yview)
        self.preview1['yscrollcommand'] = self.preview1ScrollY.set
        self.preview1ScrollY.pack(fill='y', side='right')
        self.preview1.pack()
        self.preview1ScrollX.pack(fill='x')
        self.preview1['state'] = 'disabled'

        # preview2
        # self.preview2Label = ttk.Label(self.tesseractxmlFrame, text=TESSERRACT_OCR_PREVIEW_LABEL)
        # self.preview2Label.pack()

        self.preview2Var = ttk.StringVar()
        self.preview2Var.set('MuPdf xml')
        self.preview2Var.trace_add('write', lambda a,b,c : 
                                    self.controller.get_preview(self.pageInd, 
                                                                self.previewTypes[self.preview1Var.get()],
                                                                self.previewTypes[self.preview2Var.get()]))
        self.preview2Combobox = Combobox(self.tesseractxmlFrame,textvariable=self.preview2Var)
        self.preview2Combobox['values'] = list(self.previewTypes.keys())
        self.preview2Combobox.pack()

        self.preview2 = ttk.Text(self.tesseractxmlFrame, width=80, height=37, wrap=ttk.NONE)
        self.preview2ScrollX = ttk.Scrollbar(self.tesseractxmlFrame,
                                                 orient='horizontal',
                                                 command=self.preview2.xview)
        self.preview2['xscrollcommand'] = self.preview2ScrollX.set
        self.preview2ScrollY = ttk.Scrollbar(self.tesseractxmlFrame,
                                                 orient='vertical',
                                                 command=self.preview2.yview)
        self.preview2['yscrollcommand'] = self.preview2ScrollY.set
        self.preview2ScrollY.pack(fill='y', side='right')
        self.preview2.pack()
        self.preview2ScrollX.pack(fill='x')
        self.preview2['state'] = 'disabled'

        # color tags for xml views
        self.preview1.tag_configure("orange", foreground="orange")
        self.preview1.tag_configure("blue", foreground="blue")
        self.preview1.tag_configure("green", foreground="green")
        self.preview1.tag_configure("red", foreground="red")
        self.preview1.tag_configure("lightblue", foreground="#00D4FF")
        self.preview1.tag_configure("black", foreground="#000000")

        self.preview2.tag_configure("orange", foreground="orange")
        self.preview2.tag_configure("blue", foreground="blue")
        self.preview2.tag_configure("green", foreground="green")
        self.preview2.tag_configure("red", foreground="red")
        self.preview2.tag_configure("lightblue", foreground="#00D4FF")
        self.preview2.tag_configure("black", foreground="#000000")

        self.pageInd = 0

    def __click_open_file(self):
        """
        Command for open file button
        """
        filename = filedialog.askopenfilename(title=OPEN_FILE_DIALOG,
                                              filetypes=(("pdf files", "*.pdf"), ("all files", "*.*")))
        self.inputButton['state'] = 'disabled'
        if filename != "":
            if not filename.endswith('.pdf'):
                ttk.messagebox.showinfo(title='Hibás kiterjesztés', message='A megnyitott fájl nem egy pdf.')
                self.inputButton['state'] = 'normal'
                return

            self.openFilename = filename
            self.controller.set_input_path(self.openFilename)
            # threading.Thread(target=self.controller.set_input_path,
            #                  args=[self.openFilename])
            thread = threading.Thread(target=self.__fastest_open_pdf_view,
                                      daemon=True)
            thread.start()

    def __open_pdf_view(self):
        """
        creates a preview of the pdf
        """
        self.pdfPages = []
        pdf_file = fitz.open(self.openFilename)
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            image_list = page.get_images()

            for image_index, img in enumerate(image_list, start=1):
                xref = img[0]

                base_image = pdf_file.extract_image(xref)
                # image_bytes = base_image["image"]

                image = Image.open(io.BytesIO(base_image["image"])).resize((400, 600))
                image = ImageTk.PhotoImage(image)
                self.pdfPages.append(image)
            # print(page_index)save_typePDF

        self.pageInd = 0
        self.pdfSpinBox['from_'] = 1
        self.pdfSpinBox['to'] = len(pdf_file)
        self.pageIndVar.set(str(self.pageInd + 1))
        self.pdfView.config(image=self.pdfPages[self.pageInd])
        self.controller.get_preview(self.pageInd, self.previewTypes[self.preview1Var.get()],self.previewTypes[self.preview2Var.get()])

    def __fast_open_pdf_view(self):
        self.pdfPages = []
        pdf_file = fitz.open(self.openFilename)

        for page in pdf_file:
            pix = page.get_pixmap()
            pix1 = fitz.Pixmap(pix,0) if pix.alpha else pix
            img = pix1.tobytes("ppm")#getImageData("ppm")
            image = Image.open(io.BytesIO(img)).resize((400, 600))
            image = ImageTk.PhotoImage(image)
            self.pdfPages.append(image)

        self.pageInd = 0
        self.pdfSpinBox['from_'] = 1
        self.pdfSpinBox['to'] = len(pdf_file)
        self.pageIndVar.set(str(self.pageInd + 1))
        self.pdfView.config(image=self.pdfPages[self.pageInd])
        self.controller.get_preview(self.pageInd, self.previewTypes[self.preview1Var.get()],self.previewTypes[self.preview2Var.get()])     

    def __fastest_open_pdf_view(self):
        self.pdfPages = []
        pdfPages = self.controller.get_model_images()

        for page in pdfPages:
            image = page.resize((400,600))
            image = ImageTk.PhotoImage(image)
            self.pdfPages.append(image)

        self.pageInd = 0
        self.pdfSpinBox['from_'] = 1
        self.pdfSpinBox['to'] = len(pdfPages)
        self.pageIndVar.set(str(self.pageInd + 1))
        self.pdfView.config(image=self.pdfPages[self.pageInd])
        self.controller.get_preview(self.pageInd, self.previewTypes[self.preview1Var.get()],self.previewTypes[self.preview2Var.get()])     

    def __click_save_file(self):
        """
        Command for save file button
        Opens a filedialog to save the file
        """
        self.saveFilename = filedialog.asksaveasfilename(title=SAVE_FILE_DIALOG,
                                                         filetypes=(("xml files", "*.xml"), ("all files", "*.*")))
        self.controller.set_output_path(self.saveFilename)

    def __start_processing_command(self):
        """
        Runs the given model
        """
        self.controller.run_model(process_type=self.processType,
                                  load_mode_type=self.loadModeVar,
                                  save_type=self.saveType,
                                  layoutRecognize=self.layoutRecognize,
                                  poemRecognize=self.poemRecognize)

    def __click_help(self):
        help_window = ttk.Toplevel(self)
        help_window.title(HELP_WINDOW_NAME)
        help_window.geometry("1850x900")
        text = ttk.Text(help_window, width=220, height=55)
        text.insert(ttk.END, HELP_TEXT)
        text['state'] = 'disabled'
        text.pack()

    def set_process_info(self, process_info):
        """
        method for controller, it can display information about the status
        """
        self.processInfo.config(text=process_info)

    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        """
        self.controller = controller

    def message_box(self, message, title=None):
        """
        Opens a messagebox with the given message and title
        """
        ttk.messagebox.showinfo(title=title, message=message)

    def __click_back_button_command(self):
        """
        Step back to the previous page in the pdf view
        """
        if self.pageInd > 0:
            self.pageInd -= 1
            self.pageIndVar.set(str(self.pageInd + 1))
        self.pdfView.config(image=self.pdfPages[self.pageInd])
        self.controller.get_preview(self.pageInd, self.previewTypes[self.preview1Var.get()],self.previewTypes[self.preview2Var.get()])

    def __click_forward_button_command(self):
        """
        Step forward to the next page in the pdf view
        """
        # self.buttonForward['state'] = 'disabled'
        if self.pageInd < len(self.pdfPages) - 1:
            self.pageInd += 1
            self.pageIndVar.set(str(self.pageInd + 1))
        self.pdfView.config(image=self.pdfPages[self.pageInd])
        self.controller.get_preview(self.pageInd, self.previewTypes[self.preview1Var.get()],self.previewTypes[self.preview2Var.get()])
        # self.buttonForward['state'] = 'normal'

    def update_xml_preview(self, viewstr: str):
        """
        Updates the syntax highlight of the preview
        :param viewstr: str
        """
        xml_view = self.preview1
        if viewstr == 'TESSERACT':
            xml_view = self.preview2

        start = "1.0"
        end = "end"
        xml_view['state'] = 'normal'

        color_dict = {
            'Page': 'green',
            'PrintSpace': 'green',
            'ComposedBlock': 'green',
            'TextBlock': 'green',
            'TextLine': 'blue',
            'String': 'red',
            'HPOS': 'lightblue',
            'VPOS': 'lightblue',
            'WIDTH': 'lightblue',
            'HEIGHT': 'lightblue',
            'ID': 'lightblue',
            'CONTENT': 'lightblue',
            'PHYSICAL_IMG_NR': 'lightblue',
            'STYLE' : 'lightblue',
            'TYPE' : 'lightblue',
            '<title>' : 'red',
            '<paragraph>' : 'green',
            '<page_number>' : 'lightblue',
        }
        for word in color_dict:
            xml_view.mark_set('SearchFrom', start)
            xml_view.mark_set('SearchTo', start)
            xml_view.mark_set('SearchLimit', end)

            word_length = ttk.IntVar()
            while True:
                index = xml_view.search(word, 'SearchTo', 'SearchLimit', count=word_length, regexp=False)

                if index == '':
                    break
                if word_length.get() == 0:
                    break

                xml_view.mark_set('SearchFrom', index)
                xml_view.mark_set('SearchTo', '%s+%sc' % (index, word_length.get()))
                xml_view.tag_add(color_dict[word], 'SearchFrom', 'SearchTo')

        word_length = ttk.IntVar()

        regex_string = [r'".*"', r"'.*'"]

        for pattern in regex_string:
            xml_view.mark_set('SearchFrom', start)
            xml_view.mark_set('SearchTo', end)

            num = int(regex_string.index(pattern))

            while True:
                index = xml_view.search(pattern, 'SearchFrom', 'SearchTo', count=word_length, regexp=True)

                if index == '':
                    break

                if num == 1:
                    xml_view.tag_add('black', index, index + 'lineend')
                elif num == 0:
                    xml_view.tag_add('orange', index, '%s+%sc' % (index, word_length.get()))

                xml_view.mark_set('SearchFrom', '%s+%sc' % (index, word_length.get()))

        xml_view['state'] = 'disabled'


class Controller:
    """
    Class to connect the Model and the View
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.preview_thread = None
        self.model_thread = None

    def set_input_path(self, input_path):
        """
        Set the selected input path in the Model and, display it on the View.
        :param input_path: str
        """
        try:
            print('input_path',input_path)
            self.model.set_input_path(input_path)
            self.view.inputEntry.delete(0, ttk.END)
            self.view.inputEntry.insert(ttk.END, input_path)
        except Exception as e:
            ttk.messagebox.showinfo(title=None, message="Hiba a megnyitáskor " + str(e))

    def set_output_path(self, output_path):
        """
        Set the selected output path in the Model and, display it on the View.
        :param output_path: str
        """
        try:
            self.model.set_output_path(output_path)

            self.view.outputEntry.delete(0, ttk.END)
            self.view.outputEntry.insert(ttk.END, output_path)
        except Exception as e:
            ttk.messagebox.showinfo(title=None, message="Hiba a kimenet beállításakor" + str(e))

    def run_model(self, process_type, load_mode_type, save_type, layoutRecognize, poemRecognize):
        """
        Runs the model
        :param process_type: StringVar, (PYMUPDF or TESSERACT or TESSERACT_S)
        :param load_mode_type: StringVar
        :param save_type: StringVar
        """
        try:
            self.view.progBar['value'] = 0
            self.model_thread = threading.Thread(target=self.model.run,
                                                 args=[process_type.get(), load_mode_type.get(), save_type.get(), layoutRecognize.get(), poemRecognize.get()],
                                                 daemon=True)
            self.model_thread.start()

            # ttk.messagebox.showinfo(title=None, message="Kész")
        except Exception as e:
            ttk.messagebox.showinfo(title=None, message="Hiba a feldolgozás közben" + str(e))

    def page_processed(self, page_count, current_page):
        """
        Display the current status of the processing
        :param page_count: int, number of pages in the pdf
        :param current_page: int, number of pages that are finished
        """
        self.view.progBar['value'] = (current_page / page_count) * 100
        # self.view.processInfo['text'] = "Tesseract Ocr feldolgozás " +  str(currentPage) +"/" +str(pageCount)

    def set_process_info(self, process_info):
        """
        Display information about the processing
        """
        self.view.set_process_info(process_info)

    def set_xml_preview(self, pymu_str: str, tesseract_str: str):
        """
        Sets the content of the xml previews on the View
        :param pymu_str: str, content of the PyMuPdf preview
        :param tesseract_str: str content of the Tesseract ocr preview
        """
        self.view.preview1['state'] = 'normal'
        self.view.preview2['state'] = 'normal'
        self.view.preview1.delete('1.0', ttk.END)
        self.view.preview1.insert(ttk.END, pymu_str)
        self.view.update_xml_preview('PYMUPDF')

        self.view.preview2.delete('1.0', ttk.END)
        self.view.preview2.insert(ttk.END, tesseract_str)
        self.view.update_xml_preview('TESSERACT')
        self.view.inputButton['state'] = 'normal'
        self.view.preview1['state'] = 'disabled'
        self.view.preview2['state'] = 'disabled'

    def get_preview(self, page_number: int, preview1_type, preview2_type):
        """
        Starts to processing the xml previews of the given page
        :param page_number: int
        """
        # self.model.get_preview(pageNumber)
        self.set_xml_preview(LOADING, LOADING)
        # thread =
        self.preview_thread = threading.Thread(target=self.model.get_preview, args=[page_number, preview1_type, preview2_type], daemon=True).start()

    def send_message(self, message, title=None):
        self.view.message_box(message, title)

    def set_running(self, is_running):
        if is_running:
            self.view.startButton['state'] = 'disabled'
        else:
            self.view.startButton['state'] = 'normal'

    def get_model_images(self):
        return self.model.get_images()

class PdfToXmlApp(ttk.Tk):
    """
    Class for the UI PdfToXml Application
    """
    def __init__(self):
        super().__init__()

        self.title('PdfToXml App')
        self.geometry("1850x900")
        self.model = ModelForView()
        #self.model.set_convert_pdf_to_images(PyMuPdfModule(self.model))
        self.model.set_convert_pdf_to_images(FastImgLoad(self.model))

        self.view = View(self)
        self.view.grid(row=0, column=0, padx=10, pady=10)

        self.controller = Controller(self.model, self.view)

        self.view.set_controller(self.controller)
        self.model.set_controller(self.controller)

    def on_close_app(self):
        """
        Command for the close the application
        """
        if self.controller.model_thread is not None and self.controller.model_thread.is_alive():
            if ttk.messagebox.askokcancel('Bezárás',
                                          'A konvertálás még fut! Biztosan megszakítja és bezárja a programot?'):
                self.model.stop_all_process()
                self.destroy()

        else:
            self.destroy()


if __name__ == '__main__':
    app = PdfToXmlApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close_app)
    app.mainloop()
    a = ModelForView()
