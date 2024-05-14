# Delta_B saturation to avoid too big prediction offset
def Saturate(Pred_Delta_B, Pred_Saturation, Delta_B_treshold):
    if Pred_Saturation == True:
        if Pred_Delta_B > Delta_B_treshold:
            Pred_Delta_B = Delta_B_treshold
        elif Pred_Delta_B < -Delta_B_treshold:
            Pred_Delta_B = -Delta_B_treshold
    return Pred_Delta_B
