TARGET_NAMES_CELEBA = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
    "Skin_Color"
]

TARGET_NAMES_UTK_DATASET = [
    "Age",
    "Gender",
    "Skin_Color"
]

PATHS = {
    'DiffaePaths@SEM@CLIP-Batch_size2-ellWGS-eps0.1_0.2-learnSV-learnGammas-Beta0.7-r-sim-contrastive-tau_0.07+10.0xID+3.0xLPIPS-SGD-lr_0.01-iter_40000@attributes_final_celebA': {
        'eps': 0.05,
        TARGET_NAMES_CELEBA.index("Young"): [ # Age
            {'k': 0, 'direction':1, 'range': [30,35]} ,  # old
            {'k': 0, 'direction':-1, 'range': [30,35]}   # young
        ],
        TARGET_NAMES_CELEBA.index("Skin_Color"): [ 
            {'k': 10, 'direction':1, 'range': [20,33]},   
            {'k': 12, 'direction':1, 'range': [25,35]}, # dark skin
        ],
    },
    'DiffaePaths@SEM@CLIP-Batch_size2-ellWGS-eps0.1_0.2-learnSV-learnGammas-Beta0.7-r-sim-contrastive-tau_0.07+10.0xID+3.0xLPIPS-SGD-lr_0.01-iter_40000@attributes_final_utkFace': {
        'eps': 0.05,
        TARGET_NAMES_UTK_DATASET.index("Age"): [ # Age
            {'k': 0, 'direction':-1, 'range': [30,40]},   # young
            {'k': 0, 'direction':1, 'range': [25,30]},  # old
        ],
        TARGET_NAMES_UTK_DATASET.index("Skin_Color"): [
            {'k': 12, 'direction':-1, 'range': [25,30]},   # pale skin
            {'k': 12, 'direction':1, 'range': [20,40]}, # dark skin
        ],
    },
}