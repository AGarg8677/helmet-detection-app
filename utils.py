def match_helmets_to_people(detections, helmet_label, person_label, iou_threshold=0.3):
    persons = [d for d in detections if d['class'] == person_label]
    helmets = [d for d in detections if d['class'] == helmet_label]

    matched = set()
    for person in persons:
        px1, py1, px2, py2 = person['box']
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet['box']
            if bbox_iou((px1, py1, px2, py2), (hx1, hy1, hx2, hy2)) > iou_threshold:
                matched.add(person['id'])
                break
    # Return True if any person is without helmet
    return len(persons) > len(matched)

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

