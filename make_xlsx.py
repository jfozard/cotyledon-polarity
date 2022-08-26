
import xlsxwriter
from PIL import Image
import io
import numpy as np

def buffer_image(A, format='PNG'):
    # https://stackoverflow.com/questions/66776526/scale-images-in-a-readable-manner-in-xlsxwriter
    # Store image in buffer, so we don't have to write it to disk.
#    print('buffer image', A.shape, A.dtype, np.max(A), np.min(A))
    image = Image.fromarray(A.astype(np.uint8))
    buffer_image = io.BytesIO()
    image.save(buffer_image, format=format)
    return buffer_image, image


class TextCell(object):
    def __init__(self, text):
        self.text = text

    def write(self, worksheet, row, col):
        worksheet.write(row, col, self.text)
        return row + 1, col + 1
    
class ImageCell(object):
    def __init__(self, image, style_name=None):
        self.image = image
        self.style_name = style_name

    def write(self, worksheet, row, col):
        
        image_buffer, image = buffer_image(self.image)

        R = 80
        
        h = worksheet.row_height_tmp.get(row, 10)
        if h<R:#self.image.shape[0]:
            worksheet.row_height_tmp[row] = R#self.image.shape[0]
        w = worksheet.col_width_tmp.get(col, 10)
        if w<R:#self.image.shape[1]:
            worksheet.col_width_tmp[col] = R#self.image.shape[1]
        s = R/max(self.image.shape[0:2])
        data = {'x_scale': s, 'y_scale': s, 'object_position': 2}
        if self.style_name:
            worksheet.insert_image(row, col, '', {'image_data': image_buffer, **data} )
            worksheet.write_blank(row, col, None, worksheet.formats[self.style_name] )
        else:
            worksheet.insert_image(row, col, '', {'image_data': image_buffer, **data} )
        return row + 1, col + 1
        
class ValueCell(object):
    def __init__(self, value):
        self.value = value

    def write(self, worksheet, row, col):
        worksheet.write(row, col, self.value)
        return row + 1, col + 1

class BlankCell(object):
    def __init__(self):
        pass
    
    def write(self, worksheet, row, col):
        return row + 1, col + 1

class CellBlock(object):
    def __init__(self, rows, cols, data):
        self.rows = rows
        self.cols = cols
        self.data = data

    def write(self, worksheet, row, col=0):
        print('Cell block', self.rows, self.cols, len(self.data))
        pos_row = row
        new_pos_row = row
        new_pos_col = col
        for i in range(self.rows):
            pos_col = col
            for j in range(self.cols):
                if i*self.cols + j < len(self.data) and self.data[i*self.cols +j] is not None:
                    nr, pos_col = self.data[i*self.cols + j].write(worksheet, pos_row, pos_col)
                    new_pos_row = max(new_pos_row, nr)
                    
            new_pos_col = max(pos_col, new_pos_col)
            pos_row = new_pos_row
        return new_pos_row, new_pos_col
        
def make_xlsx(filename, data_list, headers=None):
    workbook = xlsxwriter.Workbook(filename)


    bold = workbook.add_format({'bold': True})
                          
    
        
    for i, group in enumerate(data_list):
        worksheet = workbook.add_worksheet(f'Data S{i+1}')

        worksheet.row_height_tmp = { 0:100}
        worksheet.col_width_tmp = { 0:100}

        worksheet.formats = { 'blue': workbook.add_format({'border':5, 'border_color': '#0000FF'}),
                          'purple': workbook.add_format({'border':5, 'border_color': '#FF00FF'}),
                          'red_t0':  workbook.add_format({'left':5, 'left_color': '#FF0000',
                                                          'top':5, 'top_color': '#FF0000',
                                                          'bottom':5, 'bottom_color': '#FF0000'}),
                          'red_t1':  workbook.add_format({'right':5, 'right_color': '#FF0000',
                                                          'top':5, 'top_color': '#FF0000',
                                                          'bottom':5, 'bottom_color': '#FF0000'}),
        }

        c_row = 0

        if headers:
            worksheet.write(c_row, 0, headers[i], bold)
            c_row += 2
            
        for panel in group:
            for d in panel:
                c_row, c_col = d.write(worksheet, c_row, 0)
            c_row += 1

        for row, v in worksheet.row_height_tmp.items():
            worksheet.set_row_pixels(row, v+5)

        for col in range(max(worksheet.col_width_tmp)+1):
            v = worksheet.col_width_tmp.get(col, 20)
            worksheet.set_column_pixels(col, col, v+5)
#        print('set_column_pixels', col, v)

    workbook.close()
    
        
        
