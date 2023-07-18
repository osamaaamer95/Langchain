import os

from Projects.PDFReader.pdf_reader import summarize_pdf

#####
# PDF Summarizer
#####

# get path to pdf file
dir_path = os.path.dirname(os.path.realpath(__file__))
# summarize PDF
summarize_pdf(dir_path + "/PDFReader/resume.pdf")
