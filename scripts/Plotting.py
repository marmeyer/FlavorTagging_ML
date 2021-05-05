import ROOT as rt

rt.gROOT.SetBatch(rt.kTRUE)
rt.gStyle.SetOptStat(0);

f_b = rt.TFile("../training_data/input/b.root","READ")
f_c = rt.TFile("../training_data/input/c.root","READ")
f_uds = rt.TFile("../training_data/input/uds.root","READ")


features = {"trk1d0sig" : [300, 0.0001, 300, True], 
            "trk2d0sig" : [250, 0.0001, 250, True],
            "trk1z0sig" : [250, 0.0001, 250, True],
            "trk2z0sig" : [250, 0.0001, 250, True],
            "trk1pt_jete" : [100, 0, 0.5, True],
            "trk2pt_jete" : [100, 0, 0.5, True],
            "jprobr25sigma" : [100, 0.0001, 1, True],
            "jprobz25sigma" : [100, 0.0001, 1, True],
            "d0bprob2" : [100, 0, 1, True],
            "d0cprob2" : [100, 0, 1, True],
            "d0qprob2" : [100, 0, 1, True],
            "z0bprob2" : [100, 0, 1, True],
            "z0cprob2" : [100, 0, 1, True],
            "z0qprob2" : [100, 0, 1, True],
            "nmuon" : [3, 0, 3, False],
            "nelectron" : [3, 0, 3, False],
            "trkmass" : [100, 0, 10, True],
            "jprobr2" : [100, 0.0001, 1, True],
            "jprobz2" : [100, 0.0001, 1, True],
            "vtxlen1_jete" : [100, 0, 0.5, True],
            "vtxsig1_jete" : [100, 0.0001, 1, True],
            "vtxdirang1_jete" : [100, 0.0001, 10, True],
            "vtxmom1_jete" : [100, 0.0001, 1, True],
            "vtxmass1" : [100, 0, 5, True],
            "vtxmult1" : [10, 0, 10, False],
            "vtxmasspc" : [100, 0, 10, True],
            "vtxprob" : [100, 0.0001, 1, True],
            "1vtxprob" : [100, 0.0001, 1, True],
            "vtxlen12all_jete" : [100, 0, 0.1, True],
            "vtxmassall" : [100, 0, 10, True],
            "vtxlen2_jete" : [100, 0, 0.5, True],
            "vtxsig2_jete" : [100, 0.0001, 10, True],
            "vtxdirang2_jete" : [100, 0.0001, 10, True],
            "vtxmom2_jete" : [100, 0.0001, 1, True],
            "vtxmass2" : [100, 0, 5, True],
            "vtxmult2" : [10, 0, 10, False],
            "vtxlen12_jete" : [100, 0, 0.1, True],
            "vtxsig12_jete" : [100, 0.0001, 5, True],
            "vtxdirang12_jete" : [100, 0.0001, 10, True],
            "vtxmom_jete" : [100, 0.0001, 1, True],
            "vtxmass" : [100, 0, 5, True],
            "vtxmult" : [10, 0, 10, False],
            "nvtx" : [4, 0, 4, False],
            "nvtxall" : [4, 0, 4, False]
           }


for feature in features:
    c = rt.TCanvas("c_"+feature, "canvas", 800, 800)
    
    h_b = rt.TH1F("b_"+feature, "b_"+feature, features[feature][0],features[feature][1], features[feature][2])
    h_c = rt.TH1F("c_"+feature, "c_"+feature, features[feature][0],features[feature][1], features[feature][2])
    h_uds = rt.TH1F("uds_"+feature, "uds_"+feature, features[feature][0],features[feature][1], features[feature][2])
    
    tree_b = f_b.Get("ntp")
    tree_c = f_c.Get("ntp")
    tree_uds = f_uds.Get("ntp")
    
    tree_b.Draw(feature+'>>b_'+feature)
    tree_c.Draw(feature+'>>c_'+feature)
    tree_uds.Draw(feature+'>>uds_'+feature)
    
    h_b.Scale(1/h_b.Integral())
    h_c.Scale(1/h_c.Integral())
    h_uds.Scale(1/h_uds.Integral())
    
    max_b = h_b.GetMaximum()
    max_c = h_c.GetMaximum()
    max_uds = h_uds.GetMaximum()
    
    if (max_b > max_c and max_b > max_uds):
        max = max_b
    elif (max_c > max_b and max_c > max_uds):
        max = max_c
    elif (max_uds > max_b and max_uds > max_c):   
        max = max_uds
        
    h_b.GetYaxis().SetRangeUser(0,max*1.1)
    
    h_b.SetLineColor(rt.kRed)
    h_c.SetLineColor(rt.kGreen)
    h_uds.SetLineColor(rt.kBlue)
    

    if (features[feature][3] == True):
        if (feature=="jprobz25sigma" or feature=="vtxdirang12_jete" or feature=="vtxprob" or feature=="vtxsig1_jete"):
            h_b.GetYaxis().SetRangeUser(0.001,max*1.1)
        else:
            h_b.GetYaxis().SetRangeUser(0.0001,max*1.1)
        c.SetLogy()
    
    h_b.Draw("hist")
    h_c.Draw("same hist")
    h_uds.Draw("same hist")
    h_b.SetTitle(feature)
    
    legend = rt.TLegend(0.62,0.65,0.86,0.87);
    legend.SetTextFont(42);
    legend.AddEntry(h_b, "b jets", "l");
    legend.AddEntry(h_c, "c jets", "l");
    legend.AddEntry(h_uds, "light jets", "l");
    legend.SetBorderSize(0)
    legend.Draw("same")
    
    c.Print("figures/"+feature+".png")