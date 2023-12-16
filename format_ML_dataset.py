#!/usr/bin/python3

import sys
import csv

bioactivity = sys.argv[1]
apdps = sys.argv[2]
outp = sys.argv[3]


# resulting dataset should have 194,824 samples
with open(bioactivity, 'r') as bio:
    moles = bio.readlines()
    moles.pop(0)

bio_fingers = {}

for mole in moles:
    info = mole.split(',')
    zinc_id = info[1]
    tranch = info[2]
    del info[:3]
    
    bio_finger = "".join(info)

    bio_finger = bio_finger[:-1]

    bio_fingers[zinc_id] = [tranch, bio_finger]

with open(apdps, 'r') as mole_fings:
    moles = mole_fings.readlines()
    moles.pop(0)

mole_fingers = {}

for mole in moles:
    info = mole.split(',')
    zinc_id = info[1]

    del info[:2]
        
    for i in range(len(info)):
        info[i] = str(int(float(info[i])))

    mole_finger = "".join(info)

    mole_finger = mole_finger[:-1]

    mole_fingers[zinc_id] = mole_finger

outp_lines = [['zinc_id', 'tranch', 'apdp', 'bioactivity']]

for zinc_id in bio_fingers.keys():
    o_line = []

    tranch = str(bio_fingers[zinc_id][0])
    bio_finger = str(bio_fingers[zinc_id][1])
    mole_finger = str(mole_fingers[zinc_id])
    
    o_line.append([str(zinc_id), tranch, mole_finger, bio_finger])

    outp_lines.append(o_line)

with open(outp, 'w') as data_file:
    writer = csv.writer(data_file, delimiter='\t', lineterminator='\n')
    for o_line in outp_lines:
        writer.writerow(o_line)

