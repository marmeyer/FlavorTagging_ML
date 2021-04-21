import torch
import torch.nn as nn
import torch.nn.functional as F
import helperfunctions
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import FullyConnectedNN_train
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self, input_nodes):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_nodes, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.sigmoid(self.fc5(x))
        x=self.fc5(x)
        return x
    

if __name__=="__main__":
        
    inputdata= {
        "input_path_b" : '/home/mameyer/FlavorTag_MachineLearning/training_data/output/data_b.h5',
        "input_path_c" : '/home/mameyer/FlavorTag_MachineLearning/training_data/output/data_c.h5',
        "input_path_uds" : '/home/mameyer/FlavorTag_MachineLearning/training_data/output/data_uds_checked.h5',
    }
    
    # trk1d0sig has to be the first of the listed variables, preselection done (trk1d0sig!=0)
    parameters = {
        "cat1" : {
                "do_training"  : False,
                "model_name"   : 'training_4layers_cat1_withoutweights_notnormalized',
                "features"     : ["trk1d0sig", "trk2d0sig", "trk1z0sig", "trk2z0sig", "trk1pt_jete", "trk2pt_jete",                                                 "jprobr25sigma", "jprobz25sigma", "d0bprob2", "d0cprob2", "d0qprob2", "z0bprob2", "z0cprob2",                                     "z0qprob2", "nmuon", "nelectron", "trkmass"],
                "batch_size"   : 50,
                "N_epochs"     : 50
        },
        "cat2" : {
                "do_training"  : True,
                "model_name"   : 'testtraining_cat2',
                "features"     : ["trk1d0sig", "trk2d0sig", "trk1z0sig", "trk2z0sig", "trk1pt_jete", "trk2pt_jete", "jprobr2",                                     "jprobz2", "vtxlen1_jete", "vtxsig1_jete", "vtxdirang1_jete", "vtxmom1_jete", "vtxmass1",                                         "vtxmult1", "vtxmasspc", "vtxprob", "d0bprob2", "d0cprob2", "d0qprob2", "z0bprob2", "z0cprob2",                                   "z0qprob2", "trkmass", "nelectron", "nmuon"],
                "batch_size"   : 50,
                "N_epochs"     : 50
        },
        "cat3" : {
                "do_training"  : True,
                "model_name"   : 'testtraining_cat3',
                "features"     : ["trk1d0sig", "trk2d0sig", "trk1z0sig", "trk2z0sig", "trk1pt_jete", "trk2pt_jete", "jprobr2",                                     "jprobz2", "vtxlen1_jete", "vtxsig1_jete", "vtxdirang1_jete", "vtxmom1_jete", "vtxmass1",                                         "vtxmult1", "vtxmasspc", "vtxprob","1vtxprob", "vtxlen12all_jete", "vtxmassall"],
                "batch_size"   : 50,
                "N_epochs"     : 50
        },
        "cat4" : {
                "do_training"  : False,
                "model_name"   : 'training_4layers_cat4_withoutweights_150nodes_100epochs',
                "features"     : ["trk1d0sig", "trk2d0sig", "trk1z0sig", "trk2z0sig", "trk1pt_jete", "trk2pt_jete", "jprobr2",                                     "jprobz2","vtxlen1_jete", "vtxsig1_jete", "vtxdirang1_jete", "vtxmom1_jete", "vtxmass1",                                         "vtxmult1", "vtxmasspc", "vtxprob", "vtxlen2_jete", "vtxsig2_jete", "vtxdirang2_jete",                                           "vtxmom2_jete", "vtxmass2", "vtxmult2", "vtxlen12_jete", "vtxsig12_jete", "vtxdirang12_jete",                                     "vtxmom_jete", "vtxmass", "vtxmult", "1vtxprob"],
                "batch_size"   : 50,
                "N_epochs"     : 100
        }
        }
        
    #read HDF5 input data sets
    print('Number of b jets / c jets / uds jets : ')
    data_b = helperfunctions.HDF5Dataset(inputdata["input_path_b"])
    data_c = helperfunctions.HDF5Dataset(inputdata["input_path_c"])
    data_uds = helperfunctions.HDF5Dataset(inputdata["input_path_uds"])
    data = [data_b,data_c,data_uds]   
        
    #convert input data sets into torch tensor, do selection for categories
    data_tensor_cat = {}
    #categories = ["cat1", "cat2", "cat3", "cat4"]
    categories = ["cat4"]
    for cat in categories:
        data_tensor = helperfunctions.ConvertToTensor(data,parameters[cat]["features"],cat)
        data_tensor_cat[cat] = data_tensor
        print('Category : ',  cat, ', total number of jets: ', len(data_tensor))
        print('Category : ', cat , ', number of b jets: ', len(data_tensor[data_tensor[:,len(parameters[cat]["features"])]==0 ]))
        print('Category : ', cat , ', number of c jets: ', len(data_tensor[data_tensor[:,len(parameters[cat]["features"])]==1 ]))
        print('Category : ', cat , ', number of uds jets: ', len(data_tensor[data_tensor[:,len(parameters[cat]["features"])]==2 ]))
    
    for cat in categories:
        
        data_tensor = data_tensor_cat[cat]
        
        net = Net(len(parameters[cat]["features"]))
             
        # normalize data, do not normalize target (at index len(parameters[cat]["features"]))  ## funktioniert das richtig? compare lines
        data_tensor = helperfunctions.Normalize(data_tensor, len(parameters[cat]["features"])) 

        # determine number for training, test and validation data, take into account rounding effects
        n_train,n_test,n_val = helperfunctions.Splitting(data_tensor)
    
        #split dataset in train, test, val (50%,25%,25%)
        train,val,test=torch.utils.data.random_split(data_tensor,[n_train,n_val,n_test],generator=torch.Generator().manual_seed(42))
        
        #weight all classes equally in loss , ## has number of features hard-coded, change!!!
        #samples_weight = helperfunctions.WeightClasses(train,len(parameters[cat]["features"]))
        
        trainloader = torch.utils.data.DataLoader(train, batch_size=parameters[cat]["batch_size"],shuffle=False)
        testloader  = torch.utils.data.DataLoader(test, batch_size=len(test),shuffle=False)
        validationloader = torch.utils.data.DataLoader(val, batch_size=parameters[cat]["batch_size"],shuffle=False)
    
        net.apply(helperfunctions.weights_init_uniform)
        #print(samples_weight)
        #criterion = nn.CrossEntropyLoss(samples_weight)  
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
        #inputs is needed as float, labels as long
        inputs, labels = helperfunctions.ConvertDataTypes(trainloader, len(parameters[cat]["features"]))
        
        if (parameters[cat]["do_training"]):
            FullyConnectedNN_train.train(net, inputs,labels,parameters[cat]["N_epochs"],criterion,optimizer, validationloader, len(parameters[cat]["features"]), parameters[cat]["model_name"])
            
        loss=[]
        for epoch in range(parameters[cat]["N_epochs"]):
            checkpoint = torch.load('/beegfs/desy/user/mameyer/checkpoints/' + parameters[cat]["model_name"] + '_' + str(epoch)+'.pth')
            loss.append(checkpoint['loss'])
        checkpoint = torch.load('/beegfs/desy/user/mameyer/checkpoints/'+ parameters[cat]["model_name"] + '_' + str(np.argmin(loss)) + '.pth')
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print('Loading model with minimum training loss: ', parameters[cat]["model_name"] + '_' + str(np.argmin(loss)) + '.pth')
    
        print('Number of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
        print('Only look at validation data for now!!!!!!!!!!')
    
        validationloader = torch.utils.data.DataLoader(val, batch_size=len(val),shuffle=False)
    
        correct = 0
        total = 0
        misid_casb = 0
        misid_udsasb = 0
        misid_basc = 0
        misid_udsasc = 0
        class_correct = list(0. for i in range(3))
        class_total = list(0. for i in range(3))
        with torch.no_grad():
            for data in validationloader:     ##rewrite such that function can be used
                inputs = data[:,0:len(parameters[cat]["features"])]
                inputs=inputs.float()
                labels = data[:,len(parameters[cat]["features"])]
                labels=labels.long()
                outputs=net(inputs)
            
                total, correct, class_total, class_correct, misid_casb, misid_udsasb, misid_basc, misid_udsasc = helperfunctions.DoCalcForAccuracy(total, correct, class_total, class_correct, outputs.data, labels, misid_casb, misid_udsasb, misid_basc, misid_udsasc)
           
        
        print('Accuracy of the network on the test set: ', 100 * correct / total)
        for i in range(3):
            print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
        
    # determine the ROC Curves
    
        with torch.no_grad():
            for data in validationloader:    ##rewrite such that function can be used
                inputs = data[:,0:len(parameters[cat]["features"])]
                inputs=inputs.float()
                labels = data[:,len(parameters[cat]["features"])]
                labels=labels.long()
                outputs=net(inputs)
                labels_b=np.copy(labels)
                labels_c=np.copy(labels)
                labels_uds=np.copy(labels)
                helperfunctions.PlotOutput(outputs.data[(labels_b==0),0], outputs.data[(labels_c==1),0], outputs.data[(labels_uds==2),0], 'b jets', 'c jets', 'uds jets', 'plots/'+ parameters[cat]["model_name"]+ '/'+ 'b_cat_.png')
            
                labels_roc_c_b=np.copy(labels)
                fpr1, tpr1, threshold = metrics.roc_curve(labels_roc_c_b[labels_roc_c_b!=2], outputs.data[(labels_roc_c_b!=2),1], pos_label=1)  #ctag eff, b bg
                labels_roc_c_uds=np.copy(labels)
                fpr2, tpr2, threshold = metrics.roc_curve(labels_roc_c_uds[labels_roc_c_uds!=0], outputs.data[(labels_roc_c_uds!=0),1], pos_label=1)  #ctag eff, uds bg
                helperfunctions.PlotRoc(fpr1,tpr1,fpr2,tpr2,'Ctag rate','b jets','uds jets','plots/'+ parameters[cat]["model_name"]+ '/'+ 'Roc_ctag_.png')
                                                    
                labels_roc_b_c=np.copy(labels)
                fpr3, tpr3, threshold = metrics.roc_curve(labels_roc_b_c[labels_roc_b_c!=2], outputs.data[(labels_roc_b_c!=2),0], pos_label=0)  #btag eff, c bg
                labels_roc_b_uds=np.copy(labels)
                fpr4, tpr4, threshold = metrics.roc_curve(labels_roc_b_uds[labels_roc_b_uds!=1], outputs.data[(labels_roc_b_uds!=1),0], pos_label=0)  #btag eff, uds bg
                helperfunctions.PlotRoc(fpr3,tpr3,fpr4,tpr4,'Btag rate','c jets','uds jets','plots/'+ parameters[cat]["model_name"]+ '/'+ 'Roc_btag.png')
                
                _, predicted = torch.max(outputs, 1)
                confusion_efficiency = metrics.confusion_matrix(labels,predicted,labels=None,normalize='true')
                confusion_purity = metrics.confusion_matrix(labels,predicted,labels=None,normalize='pred')
                helperfunctions.PlotConfusion(confusion_efficiency, 'plots/'+ parameters[cat]["model_name"]+ '/'+ 'confusion_eff_'+cat+'.png')
                helperfunctions.PlotConfusion(confusion_purity, 'plots/'+ parameters[cat]["model_name"]+ '/'+ 'confusion_purity_'+cat+'.png')
               
            
        validation_loss=[]
        training_loss=[]
        training_accuracy_b=[]
        training_accuracy_c=[]
        training_accuracy_o=[]
        validation_accuracy=[]
        training_accuracy=[]
        validation_accuracy_b=[]
        validation_accuracy_c=[]
        validation_accuracy_o=[]
        training_misidbasc = []
        training_misidudsasc = []
        training_misidudsasb = []
        training_misidcasb = []
        validation_misidbasc = []
        validation_misidudsasc = []
        validation_misidudsasb = []
        validation_misidcasb = []
        for epoch in range(parameters[cat]["N_epochs"]):
            checkpoint = torch.load('/beegfs/desy/user/mameyer/checkpoints/' + parameters[cat]["model_name"] + '_' + str(epoch)+'.pth')
            training_loss.append(checkpoint['loss'])
            validation_loss.append(checkpoint['loss_val'])
            training_accuracy_b.append(checkpoint['accuracy_b'])
            training_accuracy_c.append(checkpoint['accuracy_c'])
            training_accuracy_o.append(checkpoint['accuracy_o'])
            validation_accuracy.append(checkpoint['accuracy_val'])
            training_accuracy.append(checkpoint['accuracy'])
            validation_accuracy_b.append(checkpoint['accuracy_b_val'])
            validation_accuracy_c.append(checkpoint['accuracy_c_val'])
            validation_accuracy_o.append(checkpoint['accuracy_o_val'])
            training_misidbasc.append(checkpoint['midis_b_as_c'])
            training_misidudsasc.append(checkpoint['midis_uds_as_c'])
            training_misidudsasb.append(checkpoint['midis_uds_as_b'])
            training_misidcasb.append(checkpoint['midis_c_as_b'])
            validation_misidbasc.append(checkpoint['midis_b_as_c_val'])
            validation_misidudsasc.append(checkpoint['midis_uds_as_c_val'])
            validation_misidudsasb.append(checkpoint['midis_uds_as_b_val'])
            validation_misidcasb.append(checkpoint['midis_c_as_b_val'])
         
        helperfunctions.PlotLearningCurves(training_loss,validation_loss, training_accuracy, validation_accuracy, training_accuracy_b, training_accuracy_c, training_accuracy_o,validation_accuracy_b,validation_accuracy_c,validation_accuracy_o, training_misidbasc, training_misidudsasc, training_misidudsasb, training_misidcasb, validation_misidbasc, validation_misidudsasc, validation_misidudsasb, validation_misidcasb, parameters[cat]["model_name"])
    