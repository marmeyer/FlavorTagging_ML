import torch
import torch.nn as nn
import torch.nn.functional as F
import helperfunctions
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import FullyConnectedNN_train

class Net(nn.Module):

    def __init__(self, input_nodes, nodes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_nodes, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.fc3 = nn.Linear(nodes, nodes)
        self.fc4 = nn.Linear(nodes, nodes)
        self.fc5 = nn.Linear(nodes, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x=self.fc5(x)
        return x


if __name__=="__main__":
        
    inputdata= {
        "input_path_b" : '/home/mameyer/FlavorTagging_ML/training_data/output/data_b.h5',
        "input_path_c" : '/home/mameyer/FlavorTagging_ML/training_data/output/data_c.h5',
        "input_path_uds" : '/home/mameyer/FlavorTagging_ML/training_data/output/data_uds_checked.h5',
    }
    
    # trk1d0sig has to be the first of the listed variables, preselection done (trk1d0sig!=0)
    parameters = {
                "do_training"  : True,
                "model_name"   : 'OneNet_4layers_150nodes_lr0p0001_equalNumberofevents',
                "features"     : ["trk1d0sig", "trk2d0sig", "trk1z0sig", "trk2z0sig", "trk1pt_jete", "trk2pt_jete",                                                 "jprobr25sigma", "jprobz25sigma", "d0bprob2", "d0cprob2", "d0qprob2", "z0bprob2", "z0cprob2",                                     "z0qprob2", "nmuon", "nelectron", "trkmass", "jprobr2", "jprobz2", "vtxlen1_jete",                                               "vtxsig1_jete", "vtxdirang1_jete", "vtxmom1_jete", "vtxmass1", "vtxmult1", "vtxmasspc",                                           "vtxprob", "1vtxprob", "vtxlen12all_jete", "vtxmassall", "vtxlen2_jete", "vtxsig2_jete",                                         "vtxdirang2_jete","vtxmom2_jete", "vtxmass2", "vtxmult2", "vtxlen12_jete", "vtxsig12_jete",                                       "vtxdirang12_jete", "vtxmom_jete", "vtxmass", "vtxmult", "nvtx", "nvtxall"],
                "batch_size"   : 50,
                "N_epochs"     : 50,
                "N_nodes"      : 150,
                "learning_rate" : 0.0001,
                "balance_input" : True,
                "apply_sample_weights" : False,
                "use_test_data" : False
    }
    
    # initialize variables
    correct = 0
    total = 0
    misid_casb = 0
    misid_udsasb = 0
    misid_basc = 0
    misid_udsasc = 0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    
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
    
    # read HDF5 input data sets
    print('Number of b jets / c jets / uds jets : ')
    data_b = helperfunctions.HDF5Dataset(inputdata["input_path_b"])
    data_c = helperfunctions.HDF5Dataset(inputdata["input_path_c"])
    data_uds = helperfunctions.HDF5Dataset(inputdata["input_path_uds"])
    data = [data_b,data_c,data_uds]   
      
    # convert input data sets into torch tensor
    data_tensor = helperfunctions.ConvertToTensor_NoCat(data,parameters["features"])
    print('Total number of jets: ', len(data_tensor))
    print('Number of b jets: ', len(data_tensor[data_tensor[:,len(parameters["features"])]==0 ]))
    print('Number of c jets: ', len(data_tensor[data_tensor[:,len(parameters["features"])]==1 ]))
    print('Number of uds jets: ', len(data_tensor[data_tensor[:,len(parameters["features"])]==2 ]))
    
    # select equal number of b, c and light jets randomly
    if parameters["balance_input"]:
        print('Select equal number of b,c and light jets randomly')
        data_tensor_b = data_tensor[data_tensor[:,len(parameters["features"])]==0 ]
        data_tensor_c = data_tensor[data_tensor[:,len(parameters["features"])]==1 ]
        data_tensor_o = data_tensor[data_tensor[:,len(parameters["features"])]==2 ]
        data_tensor_c =data_tensor_c[torch.randint(len(data_tensor_c), (len(data_tensor_b),),     generator=torch.Generator().manual_seed(41))]
        data_tensor_o =data_tensor_o[torch.randint(len(data_tensor_o), (len(data_tensor_b),), generator=torch.Generator().manual_seed(41))]
        data_tensor = torch.cat((data_tensor_b, data_tensor_c, data_tensor_o))
         
    net = Net(len(parameters["features"]), parameters["N_nodes"])
             
    # normalize data, do not normalize target (at index len(parameters["features"]))
    data_tensor = helperfunctions.Normalize(data_tensor, len(parameters["features"])) 
    
    # determine number for training, test and validation data, take into account rounding effects
    n_train,n_test,n_val = helperfunctions.Splitting(data_tensor)
        
    # split dataset in train, test, val (50%,25%,25%)
    train,val,test=torch.utils.data.random_split(data_tensor,[n_train,n_val,n_test],generator=torch.Generator().manual_seed(42))
        
    # weight all classes equally in loss
    samples_weight = helperfunctions.WeightClasses(train,len(parameters["features"]))
    print("Sample weights are : ", samples_weight)
    
    # load training, test and validation data
    trainloader = torch.utils.data.DataLoader(train, batch_size=parameters["batch_size"],shuffle=False)
    testloader  = torch.utils.data.DataLoader(test, batch_size=len(test),shuffle=False)
    validationloader = torch.utils.data.DataLoader(val, batch_size=parameters["batch_size"],shuffle=False)
    
    # initialize weights of NN
    net.apply(helperfunctions.weights_init_uniform)
    
    # define loss function
    if parameters["apply_sample_weights"]:
        criterion = nn.CrossEntropyLoss(samples_weight)
        print('Sample weights are applied')
    else:
        criterion = nn.CrossEntropyLoss()
        print('Sample weights are not applied')
    
    # define optimizer
    optimizer = optim.Adam(net.parameters(), parameters["learning_rate"])
        
    # inputs is needed as float, labels as long
    inputs, labels = helperfunctions.ConvertDataTypes(trainloader, len(parameters["features"]))
    
    # do the training
    if (parameters["do_training"]):
        FullyConnectedNN_train.train(net, inputs, labels, parameters["N_epochs"], criterion, optimizer, validationloader, len(parameters["features"]), parameters["model_name"])
    
    # load the model with minimum training loss
    loss=[]
    for epoch in range(parameters["N_epochs"]):
        checkpoint = torch.load('/beegfs/desy/user/mameyer/checkpoints/' + parameters["model_name"] + '_' + str(epoch)+'.pth')
        loss.append(checkpoint['loss'])
    checkpoint = torch.load('/beegfs/desy/user/mameyer/checkpoints/'+ parameters["model_name"] + '_' + str(np.argmin(loss)) + '.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print('Loading model with minimum training loss: ', parameters["model_name"] + '_' + str(np.argmin(loss)) + '.pth')
    
    # print number of parameters of the NN
    print('Number of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    if parameters["use_test_data"]:
        loader = torch.utils.data.DataLoader(test, batch_size=len(test),shuffle=False)
        print('Use test data for accuracy calculation and ROC!')
    else:
        loader = torch.utils.data.DataLoader(val, batch_size=len(val),shuffle=False)
        print('Use validation data for accuracy calculation and ROC!')
        
    
    # determine accuracy in validation data 
    with torch.no_grad():
        for data in loader:     ##rewrite such that function can be used
            inputs = data[:,0:len(parameters["features"])]
            inputs=inputs.float()
            labels = data[:,len(parameters["features"])]
            labels=labels.long()
            outputs=net(inputs)
            
            total, correct, class_total, class_correct, misid_casb, misid_udsasb, misid_basc, misid_udsasc = helperfunctions.DoCalcForAccuracy(total, correct, class_total, class_correct, outputs.data, labels, misid_casb, misid_udsasb, misid_basc, misid_udsasc)
           
    print('Accuracy of the network: ', 100 * correct / total)
    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
        
    # determine ROC Curves on validation data
    with torch.no_grad():
        for data in loader:    ##rewrite such that function can be used
            inputs = data[:,0:len(parameters["features"])]
            inputs=inputs.float()
            labels = data[:,len(parameters["features"])]
            labels=labels.long()
            outputs=net(inputs)
            labels_b=np.copy(labels)
            labels_c=np.copy(labels)
            labels_uds=np.copy(labels)
            helperfunctions.PlotOutput(outputs.data[(labels_b==0),0], outputs.data[(labels_c==1),0], outputs.data[(labels_uds==2),0], 'b jets', 'c jets', 'uds jets', 'plots/'+ parameters["model_name"]+ '/'+ 'b_cat.png')
            
            labels_roc_c_b=np.copy(labels)
            fpr1, tpr1, threshold = metrics.roc_curve(labels_roc_c_b[labels_roc_c_b!=2], outputs.data[(labels_roc_c_b!=2),1], pos_label=1)  #ctag eff, b bg
            labels_roc_c_uds=np.copy(labels)
            fpr2, tpr2, threshold = metrics.roc_curve(labels_roc_c_uds[labels_roc_c_uds!=0], outputs.data[(labels_roc_c_uds!=0),1], pos_label=1)  #ctag eff, uds bg
            helperfunctions.PlotRoc(fpr1,tpr1,fpr2,tpr2,'Ctag rate','b jets','uds jets','plots/'+ parameters["model_name"]+ '/'+ 'Roc_ctag_.png')
                                                     
            labels_roc_b_c=np.copy(labels)
            fpr3, tpr3, threshold = metrics.roc_curve(labels_roc_b_c[labels_roc_b_c!=2], outputs.data[(labels_roc_b_c!=2),0], pos_label=0)  #btag eff, c bg
            labels_roc_b_uds=np.copy(labels)
            fpr4, tpr4, threshold = metrics.roc_curve(labels_roc_b_uds[labels_roc_b_uds!=1], outputs.data[(labels_roc_b_uds!=1),0], pos_label=0)  #btag eff, uds bg
            helperfunctions.PlotRoc(fpr3,tpr3,fpr4,tpr4,'Btag rate','c jets','uds jets','plots/'+ parameters["model_name"]+ '/'+ 'Roc_btag.png')
            
            # plot confusion matrices
            _, predicted = torch.max(outputs, 1)
            confusion_efficiency = metrics.confusion_matrix(labels,predicted,labels=None,normalize='true')
            confusion_purity = metrics.confusion_matrix(labels,predicted,labels=None,normalize='pred')
            helperfunctions.PlotConfusion(confusion_efficiency, 'plots/'+ parameters["model_name"]+ '/'+ 'confusion_eff.png')
            helperfunctions.PlotConfusion(confusion_purity, 'plots/'+ parameters["model_name"]+ '/'+ 'confusion_purity.png')
               
    # plot learning curves        
    for epoch in range(parameters["N_epochs"]):
        checkpoint = torch.load('/beegfs/desy/user/mameyer/checkpoints/' + parameters["model_name"] + '_' + str(epoch)+'.pth')
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
          
    
    helperfunctions.PlotLearningCurves(training_loss,validation_loss, training_accuracy, validation_accuracy, training_accuracy_b, training_accuracy_c, training_accuracy_o,validation_accuracy_b,validation_accuracy_c,validation_accuracy_o, training_misidbasc, training_misidudsasc, training_misidudsasb, training_misidcasb, validation_misidbasc, validation_misidudsasc, validation_misidudsasb, validation_misidcasb, parameters["model_name"])
     
