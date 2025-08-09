import fitz

doc = fitz.open('PDF.pdf')
print(f'PDF has {len(doc)} pages')

for i in range(len(doc)):
    page = doc[i]
    pix = page.get_pixmap()
    output_file = f'pdf_page{i+1}.png'
    pix.save(output_file)
    print(f'Page {i+1} saved as {output_file}')

print('All pages extracted successfully')