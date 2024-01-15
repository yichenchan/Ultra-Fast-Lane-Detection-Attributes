import sys
import os
import math
from xml.etree import ElementTree as etree
from xml.etree.ElementTree import Element,SubElement,ElementTree
import cv2
import numpy as np

status_list = ['normal','abnormal']
lighting_list = ['day', 'night', 'others']
weather_list = ['sunny', 'rain', 'snow', 'cloudy', 'unknown']
environmet_list = ['internal', 'closed', 'urban', 'tunnle', 'others']

# yellow黄色、white白色、red红色、blue蓝色、normal脑补无颜色
color_list = ['none', 'yellow', 'white', 'red', 'blue', 'normal']
# ptcd普通车道分隔、dxcd对向车道分隔、dlby道路边缘分隔、fjdc非机动车道分隔、yjcd应急车道分隔、gjcz公交车站分隔、gjcd公交车道分隔
funct_list = ['none', 'ptcd','dxcd','dlby','fjdc','yjcd','gjcz','gjcd']
# xx（虚线）、ss（实线）、wgx（网格线）、dlx（导流线）、tsxx（特殊虚线）、tsss（特殊实线）、dlby（道路边缘）、nbx（脑补线）
type_list = ['none', 'xx','ss','wgx','dlx','tsxx','tsss','dlby','nbx']

def process_xml(xml_path, im_path):
    print('processing ', xml_path)    

    if(not os.path.exists(im_path)):
        print("img not found:", im_path)
        return None, None

    im = cv2.imread(im_path)
    lane_label = np.zeros((im.shape[0], im.shape[1], 1))
    type_label = np.zeros((im.shape[0], im.shape[1], 1))
    func_color_label = np.zeros((im.shape[0], im.shape[1], 1))

    tree = etree.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')

    if len(objects) == 0:
        print(' error in ', xml_path)
        return np.concatenate((lane_label, type_label, func_color_label),axis=2), None

    gattrs = {}
    for obj in objects:        
        # 处理全图属性        
        if obj.find('name').text == "image_classification":                        
            attributes = obj.findall('attributes')[0].findall('attribute')
            gattrs = {}
            gattrs[attributes[0].find('name').text] = attributes[0].find('value').text
            gattrs[attributes[1].find('name').text] = attributes[1].find('value').text
            gattrs[attributes[2].find('name').text] = attributes[2].find('value').text
            gattrs[attributes[3].find('name').text] = attributes[3].find('value').text
            if gattrs['weather'] == 'night':
                gattrs['weather'] = 'unknown'
            print(gattrs)            
        else:
            pass   
    
    # 没有标注属性或者属性为非正常行驶状态，直接返回
    if len(gattrs) == 0:
        cv2.imwrite(im_path[:-4] + "_visualization.jpg", im)
        return np.concatenate((lane_label,type_label, func_color_label),axis=2), None
    elif gattrs['status'] == 'abnormal':
        cv2.putText(im, gattrs['status'], (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(im, gattrs['lighting'], (100, 300), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(im, gattrs['weather'], (100, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(im, gattrs['road_construction'], (100, 500), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.imwrite(im_path[:-4] + "_visualization.jpg", im)
        return np.concatenate((lane_label,type_label, func_color_label),axis=2), gattrs
    # 解析各条车道线信息
    else:
        cv2.putText(im, gattrs['status'], (100, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(im, gattrs['lighting'], (100, 300), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(im, gattrs['weather'], (100, 400), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        cv2.putText(im, gattrs['road_construction'], (100, 500), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)
        for obj in objects:
            name = obj.find('name').text
            if name != "image_classification" and name != "grid" and name != "guide":
                attributes = obj.findall('attributes')
                if not attributes:
                    continue
                attributes = attributes[0].findall('attribute')
                attrs = {}
                attrs[attributes[0].find('name').text] = attributes[0].find('value').text
                attrs[attributes[1].find('name').text] = attributes[1].find('value').text
                attrs[attributes[2].find('name').text] = attributes[2].find('value').text
                attrs[attributes[3].find('name').text] = attributes[3].find('value').text
                if attrs["type"] == 'nbxx':
                    attrs["type"] = 'nbx'
                point_str = obj.find('polyline').find('points').text                
                print(name,attrs, point_str, point_str.split(';'))

                lane_index = int(name) * 2 
                if (not 'combination_line' in attrs) or attrs['combination_line'] == 'normal' or attrs['combination_line'] == '0':
                    lane_index = lane_index + 1
                elif attrs['combination_line'] == '1':
                    lane_index = lane_index + 2
                else:
                    print('wrong lane index', name, attrs, "=================================")

                points = point_str.split(';')
                for i in range(len(points)-1):
                    p1 = points[i]
                    p2 = points[i+1]
                    x1, y1 = int(float(p1.split(',')[0])), int(float(p1.split(',')[1]))
                    x2, y2 = int(float(p2.split(',')[0])), int(float(p2.split(',')[1]))
                    cv2.line(lane_label, (x1,y1), (x2,y2), lane_index, 15)
                    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    funct_index = funct_list.index(attrs["function"]) 
                    color_index = color_list.index(attrs["color"])
                    func_color_index = 10 * funct_index + color_index
                    cv2.line(func_color_label, (x1,y1), (x2,y2), func_color_index, 15)

                    type_index = type_list.index(attrs["type"])                    
                    cv2.line(type_label, (x1,y1), (x2,y2), type_index, 15)
                    
        cv2.imwrite(im_path[:-4] + "_visualization.jpg", im)
        

    return np.concatenate((lane_label, type_label, func_color_label), axis=2), gattrs

label_path = sys.argv[1].rstrip('/') + '/'
image_path = sys.argv[2].rstrip('/') + '/'
xmls = os.listdir(label_path)
xmls = [xml for xml in xmls if xml.endswith('xml')]

with open('all_lane_dataset.txt', 'a', encoding='utf-8') as all_path_files:
    for xml in xmls:
        label_png, attrs = process_xml(label_path + xml, image_path + xml[:-3] + 'jpg')
        if attrs is None:
            continue
        label_png_name = image_path + xml[:-4]+str(status_list.index(attrs['status']))+str(lighting_list.index(attrs['lighting']))+str(weather_list.index(attrs['weather']))+str(environmet_list.index(attrs['road_construction'])) + '.png'
        cv2.imwrite(label_png_name, label_png)
        
        all_path_files.write(image_path + xml[:-3] + 'jpg  ' + label_png_name + '\n')


