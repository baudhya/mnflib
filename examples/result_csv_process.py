import csv

from pprint import pprint as print

data = {}

for idx in range(1, 41):
    with open(f'./snr_results/band_{idx}.csv', 'r') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        for row in reader:
            Inverse_Bands  = int(row[0])
            Percentage = float(row[1])
            LBL_next_pixel = float(row[2])
            LBL_three_pixel = float(row[3])
            MNF_next_pixel = float(row[4])
            MNF_three_pixel = float(row[5])
            MNF_improved = float(row[6])
            LBL_improved = float(row[7])

            data[(idx, Inverse_Bands)] = {
                "Inverse_Bands": Inverse_Bands,
                "Percentage": Percentage,
                "LBL_next_pixel": LBL_next_pixel,
                "LBL_three_pixel": LBL_three_pixel,
                "MNF_next_pixel": MNF_next_pixel,
                "MNF_three_pixel": MNF_three_pixel,
                "MNF_improved": MNF_improved,
                "LBL_improved": LBL_improved,
            }


improvement = {}

epsillon = 0.05

for key, row in data.items():
    b_idx, inv_band = key
    if inv_band not in improvement:
        improvement[inv_band] = {
            "MNF_improved" : 0,
            "LBL_improved" : 0
        }
    if row['LBL_improved'] > epsillon :
        improvement[inv_band]['LBL_improved'] += 1
    
    if row['MNF_improved'] > epsillon:
        improvement[inv_band]['MNF_improved'] += 1


print(improvement)
    

improvement = {}
epsillon = 1.0
for key, row in data.items():
    b_idx, inv_band = key
    if inv_band not in improvement:
        improvement[inv_band] = {
            "MNF_improved" : 0,
            "LBL_improved" : 0
        }
    if row['LBL_improved'] >= epsillon :
        print(f"LBL Band : {inv_band} : {b_idx} : {row['LBL_improved']}")
        improvement[inv_band]['LBL_improved'] += 1
    
    if row['MNF_improved'] >= epsillon:
        print(f"MNF Band : {inv_band} : {b_idx} : {row['MNF_improved']}")
        improvement[inv_band]['MNF_improved'] += 1


print(improvement)