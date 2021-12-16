import re
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg
import json

num_With_Helmet = 0
num_Without_Helmet = 0
num_WithHelmet = int()
num_WithoutHelmet = int()
withouthelmet = []
withhelmet = []

detection = []


# Zone Up Down
x_up_start = 500    #500
y_up_start = 0      #0
x_up_end = 505     #505
y_up_end = 600      #600





def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def load_config(FLAGS):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def get_center(x_center, y_center, w_certer, h_center):
    x2_center = int(w_certer / 2)
    y2_center = int(h_center / 2)
    cx_center = x_center + x2_center
    cy_center = y_center + y2_center
    # print("  test cx,cy  " + str(cx_center)+"  ,  "+ str(cy_center)) #show center point
    #cx_center = cx_center - 250  #แก้ตำแหน่งจุดเเนวนอน
    #cy_center = cy_center - 50  #แก้ตำแหน่งจุดเเนวตั้ง
    return cx_center, cy_center





def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES),
              allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values()), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    cv2.rectangle(image, (x_up_start, y_up_start),(x_up_end, y_up_end), (255, 0, 0), 3)
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    
    global num_With_Helmet 
    global num_Without_Helmet
    global num_WithHelmet 
    global num_WithoutHelmet 
    global withouthelmet 
    global withhelmet 

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes:
            continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        x_center = int(coor[1])
        y_center = int(coor[0])
        w_certer = int(coor[3])
        h_center = int(coor[2])

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        class_name = classes[class_ind]
        
        # check if class is in allowed classes
        if class_name not in allowed_classes:
            continue
        else:
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)

            # c1 แนวนอน, c2 เเนวตั้ง
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            # กรอบหมวกกันน็อค cv2.rectangle(image, พิกัดxy, พิกัดwh, bbox_color, bbox_thick)
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
            center_point001 = get_center(x_center, y_center, w_certer, h_center)  # รวมจุด
            detection.append(center_point001)
            cv2.circle(image, center_point001, 4, (255, 0, 0), 2)  # สร้างจุด
            
            if show_label:
                bbox_mess = '%s: %s' % (classes[class_ind], score)
                
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filledแถบสีตัวหนังสือ
                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
               
                
                #return(class_name)
                #if 'Without' in bbox_mess:
                #    print('Without 1')
                #elif 'With Helmet' in bbox_mess:
                #    print('With Helmet 1')

                
        # test count
   

    for (x_center, y_center) in detection:
        
        if (x_center <= x_up_end) and (x_center >= x_up_start) and (y_center <= y_up_end) and (y_center >= y_up_start) :  # เงื่อนไข ตรวจสอบ UP ZONE
            A = 'With Helmet' 
            B = 'Without Helmet'  
            C = str(bbox_mess) 
            gethelmet = {} 
            getnothelmet = {}
            if  A in C:
                cv2.rectangle(image, (x_up_start, y_up_start), (x_up_end, y_up_end), (208, 233, 40), -1)
                detection.remove((x_center, y_center))
                num_With_Helmet += 1  # บวกค่า num_up
                num_WithHelmet = num_With_Helmet
                WithHelmet = "With Helmet"':' +str(num_With_Helmet)+""
                WithoutHelmet = "Without Helmet"':'+str(num_Without_Helmet)+""
                print('With Helmet 1')
                withhelmet.append([WithHelmet,WithoutHelmet])
                with open("D:\OneDrive\Desktop\webpagetestgrahp\static\dection.json","w") as json_file:
                    json.dump(withhelmet, json_file)
               
            elif  B in C :
                cv2.rectangle(image, (x_up_start, y_up_start), (x_up_end, y_up_end), (208, 233, 40), -1)
                detection.remove((x_center, y_center))
                num_Without_Helmet += 1  # บวกค่า num_up
                num_WithoutHelmet = num_Without_Helmet
                WithHelmet =  "With Helmet"':' +str(num_With_Helmet)+""
                WithoutHelmet = "Without Helmet"':'+str(num_Without_Helmet)+""
                print('Without 1')
                withouthelmet.append([WithHelmet,WithoutHelmet]) 
                with open("D:\OneDrive\Desktop\webpagetestgrahp\static\dection.json","w") as json_file:
                    json.dump(withouthelmet, json_file)
               
    
    #cv2.rectangle(image, (20, 410), (400, 445), (255, 255, 255), -1) 
    cv2.putText(image, "with  : " + str(num_WithHelmet), (30, 450),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(image, "without : " + str(num_WithoutHelmet), (30, 495),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    #cv2.putText(image, "without : " + str(num_up), (30, 495), cv2.FONT_HERSHEY_COMPLEX, 1, (236, 236, 39), 1)
    # end test count
    return image


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou