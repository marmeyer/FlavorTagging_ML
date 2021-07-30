import ROOT as rt

rt.gROOT.SetBatch(rt.kTRUE)
rt.gStyle.SetOptStat(0);

deepjet = True

if deepjet:
    f_b = rt.TFile("../training_data/input_deepjet/b.root","READ")
    f_c = rt.TFile("../training_data/input_deepjet/c.root","READ")
    f_uds = rt.TFile("../training_data/input_deepjet/uds.root","READ")
    
                # name of the variable, number of bins, xmin, xmax, log, name of the tree
    features = {"ChargedMomentumConstituent" : [80, 0.0, 80, True, "ChargedJetConstituents"],
                "ChargedTrackMomentumFrac" : [50, 0.0, 1.0, True, "ChargedJetConstituents"],
                "ChargedTrackEnergyFrac" : [50, 0.0, 1.0, True, "ChargedJetConstituents"],
                "ChargedTrackThetaRelJet" : [100, -100.0, 100, False, "ChargedJetConstituents"],
                "ChargedTrackPtRelJet" : [100, 0.0, -100, False, "ChargedJetConstituents"],
                "ChargedTrackJetDotProduct" : [100, -50.0, 400, False, "ChargedJetConstituents"],
                "ChargedTrackJetDotProductNorm" : [100, -10.0, 60, True, "ChargedJetConstituents"],
                "ChargedDeltaRJetTrack" : [30, 0.0, 3.0, True, "ChargedJetConstituents"],
                "ChargedDeltaRJetTrackRapidity" : [30, 0.0, 3.0, True, "ChargedJetConstituents"],
                "ChargedDeltaRTrackSecVertex" : [40, 0.0, 4.0, False, "ChargedJetConstituents"],
                "ChargedTrackClosestApproachJet" : [100, 0.0, 20000, True, "ChargedJetConstituents"],
                "ChargedD0Sig" : [100, 0.0, 100, True, "ChargedJetConstituents"],
                "ChargedD0" : [50, 0.0, 50, True, "ChargedJetConstituents"],
                "ChargedZ0" : [100, 0.0, 25, True, "ChargedJetConstituents"],
                "ChargedZ0Sig" : [100, 0.0, 200, True, "ChargedJetConstituents"],
                "Charged3DImpactPar" : [50, 0.0, 100, True, "ChargedJetConstituents"],
                "Charged3DImpactParSig" : [150, 0.0, 150, False, "ChargedJetConstituents"],
                "ChargedChi2NDF" : [50, 0.0, 5, True, "ChargedJetConstituents"],
                "ChargedNDF" : [100, 0.0, 300, False, "ChargedJetConstituents"],
                "ChargedTrackUsedinPrimVertex" : [2, 0.0, 1.0, True, "ChargedJetConstituents"],
                "NeutralMomentumConstituent" : [100, 0.0, 100, True, "NeutralJetConstituents"],
                "NeutralDeltaRSecVertex" : [100, 0.0, 4.0, False, "NeutralJetConstituents"],
                "NeutralDeltaRJetTrack" : [100, 0.0, 4.0, True, "NeutralJetConstituents"],
                "NeutralDeltaRJetTrackRapidity" : [100, 0.0, 4.0, True, "NeutralJetConstituents"],
                "NeutralHCalFrac" : [50, 0.0, 1.0, True, "NeutralJetConstituents"],
                "NeutralTrackMomentumFrac" : [50, 0.0, 1.0, True, "NeutralJetConstituents"],
                "NeutralTrackEnergyFrac" : [50, 0.0, 1.0, True, "NeutralJetConstituents"],
                "JetEnergy" : [100, 0.0, 300, False, "Jets"],
                "JetPt" : [100, 0.0, 200, False, "Jets"],
                "JetMomentum" : [100, 0.0, 300, False, "Jets"],
                "JetTheta" : [100, 0, 3.5, False, "Jets"],
                "JetPhi" : [100, 0, 3.5, False, "Jets"],
                "JetRapidity" : [100, -4.0, 4.0, False, "Jets"],
                "NCharged" : [50, 0.0, 50, False, "Jets"],
                "NNeutral" : [50, 0.0, 50, False, "Jets"],
                "NSecondaryVertices" : [4, 0.0, 4, False, "Jets"],
                "NPseudoVertices" : [4, 0.0, 4, False, "Jets"],
                "SecondaryVertexMass" : [100, 0.0, 6, False, "SecondaryVertices"],
                "SecondaryVertexNTracks" : [15, 0.0, 15, False, "SecondaryVertices"],
                "SecondaryVertexChi2" : [70, 0.0, 35, True, "SecondaryVertices"],
                "SecondaryVertexChi2NDF" : [100, 0.0, 20, True, "SecondaryVertices"],
                "SecondaryVertexNDF" : [100, 0.0, 20, True, "SecondaryVertices"],
                "SecondaryVertexMomentum" : [100, 0.0, 100, False, "SecondaryVertices"],
                "SecondaryVertexEnergy" : [100, 0.0, 100, False, "SecondaryVertices"],
                "SecondaryVertexDeltaR" : [100, 0.0, 2.0, False, "SecondaryVertices"],
                "SecondaryVertexEnergyJetEnergy" : [100, 0.0, 1.0, False, "SecondaryVertices"],
                "SecondaryVertexCosMomPos" : [100, -1.0, 1.0, True, "SecondaryVertices"],
                "SecondaryVertexD0" : [100, 0.0, 100, False, "SecondaryVertices"],
                "SecondaryVertexD0Sig" : [100, 0.0, 100, False, "SecondaryVertices"],
                "SecondaryVertexZ0" : [100, 0.0, 100, True, "SecondaryVertices"],
                "SecondaryVertexZ0Sig" : [100, 0.0, 100, True, "SecondaryVertices"],
                "SecondaryVertexImpactPar3d" : [100, 0.0, 100, False, "SecondaryVertices"],
                "SecondaryVertexImpactPar3dSig" : [100, 0.0, 100, False, "SecondaryVertices"],
               }
else:
    f_b = rt.TFile("../training_data/input/b.root","READ")
    f_c = rt.TFile("../training_data/input/c.root","READ")
    f_uds = rt.TFile("../training_data/input/uds.root","READ")

    features = {"trk1d0sig" : [300, 0.0001, 300, True,"ntp"], 
                "trk2d0sig" : [250, 0.0001, 250, True,"ntp"],
                "trk1z0sig" : [250, 0.0001, 250, True,"ntp"],
                "trk2z0sig" : [250, 0.0001, 250, True,"ntp"],
                "trk1pt_jete" : [100, 0, 0.5, True,"ntp"],
                "trk2pt_jete" : [100, 0, 0.5, True,"ntp"],
                "jprobr25sigma" : [100, 0.0001, 1, True,"ntp"],
                "jprobz25sigma" : [100, 0.0001, 1, True,"ntp"],
                "d0bprob2" : [100, 0, 1, True,"ntp"],
                "d0cprob2" : [100, 0, 1, True,"ntp"],
                "d0qprob2" : [100, 0, 1, True,"ntp"],
                "z0bprob2" : [100, 0, 1, True,"ntp"],
                "z0cprob2" : [100, 0, 1, True,"ntp"],
                "z0qprob2" : [100, 0, 1, True,"ntp"],
                "nmuon" : [3, 0, 3, False,"ntp"],
                "nelectron" : [3, 0, 3, False,"ntp"],
                "trkmass" : [100, 0, 10, True,"ntp"],
                "jprobr2" : [100, 0.0001, 1, True,"ntp"],
                "jprobz2" : [100, 0.0001, 1, True,"ntp"],
                "vtxlen1_jete" : [100, 0, 0.5, True,"ntp"],
                "vtxsig1_jete" : [100, 0.0001, 1, True,"ntp"],
                "vtxdirang1_jete" : [100, 0.0001, 10, True,"ntp"],
                "vtxmom1_jete" : [100, 0.0001, 1, True,"ntp"],
                "vtxmass1" : [100, 0, 5, True,"ntp"],
                "vtxmult1" : [10, 0, 10, False,"ntp"],
                "vtxmasspc" : [100, 0, 10, True,"ntp"],
                "vtxprob" : [100, 0.0001, 1, True,"ntp"],
                "1vtxprob" : [100, 0.0001, 1, True,"ntp"],
                "vtxlen12all_jete" : [100, 0, 0.1, True,"ntp"],
                "vtxmassall" : [100, 0, 10, True,"ntp"],
                "vtxlen2_jete" : [100, 0, 0.5, True,"ntp"],
                "vtxsig2_jete" : [100, 0.0001, 10, True,"ntp"],
                "vtxdirang2_jete" : [100, 0.0001, 10, True,"ntp"],
                "vtxmom2_jete" : [100, 0.0001, 1, True,"ntp"],
                "vtxmass2" : [100, 0, 5, True,"ntp"],
                "vtxmult2" : [10, 0, 10, False,"ntp"],
                "vtxlen12_jete" : [100, 0, 0.1, True,"ntp"],
                "vtxsig12_jete" : [100, 0.0001, 5, True,"ntp"],
                "vtxdirang12_jete" : [100, 0.0001, 10, True,"ntp"],
                "vtxmom_jete" : [100, 0.0001, 1, True,"ntp"],
                "vtxmass" : [100, 0, 5, True,"ntp"],
                "vtxmult" : [10, 0, 10, False,"ntp"],
                "nvtx" : [4, 0, 4, False,"ntp"],
                "nvtxall" : [4, 0, 4, False,"ntp"]
               }


for feature in features:
    c = rt.TCanvas("c_"+feature, "canvas", 800, 800)
    
    h_b = rt.TH1F("b_"+feature, "b_"+feature, features[feature][0],features[feature][1], features[feature][2])
    h_c = rt.TH1F("c_"+feature, "c_"+feature, features[feature][0],features[feature][1], features[feature][2])
    h_uds = rt.TH1F("uds_"+feature, "uds_"+feature, features[feature][0],features[feature][1], features[feature][2])
    
    tree_b = f_b.Get(features[feature][4])
    tree_c = f_c.Get(features[feature][4])
    tree_uds = f_uds.Get(features[feature][4])
    
    tree_b.Draw(feature+'>>b_'+feature)
    tree_c.Draw(feature+'>>c_'+feature)
    tree_uds.Draw(feature+'>>uds_'+feature)
    
    h_b.Scale(1/h_b.Integral())
    h_c.Scale(1/h_c.Integral())
    h_uds.Scale(1/h_uds.Integral())
    
    max_b = h_b.GetMaximum()
    max_c = h_c.GetMaximum()
    max_uds = h_uds.GetMaximum()
  
    maximum=0
    if (max_b > max_c and max_b > max_uds):
        maximum = max_b
    elif (max_c > max_b and max_c > max_uds):
        maximum = max_c
    elif (max_uds > max_b and max_uds > max_c):   
        maximum = max_uds
    
    h_b.GetYaxis().SetRangeUser(0,maximum*1.1)
    
    h_b.SetLineColor(rt.kRed)
    h_c.SetLineColor(rt.kGreen)
    h_uds.SetLineColor(rt.kBlue)
    

    if (features[feature][3] == True):
        if (feature=="jprobz25sigma" or feature=="vtxdirang12_jete" or feature=="vtxprob" or feature=="vtxsig1_jete"):
            h_b.GetYaxis().SetRangeUser(0.001,maximum*1.1)
        else:
            h_b.GetYaxis().SetRangeUser(0.0001,maximum*1.1)
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
    
    if deepjet:
        c.Print("figures/deepjet/"+feature+".png")
    else:
        c.Print("figures/fullyconnectedNN/"+feature+".png")