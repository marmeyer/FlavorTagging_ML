import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pylab as pylab


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        
    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
            print(len(self.dataset[index]))
        return self.dataset[index]
    
def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

# function for converting hdf5 file to torch tensor if categorization is used, preselection is applied            
def ConvertToTensor(data, features, cat):
    list_datasets = []
    for dataset in data:
        list_features = []
        list_features.append(dataset["nvtx"][:])
        list_features.append(dataset["nvtxall"][:])
        for feature in features:
            list_features.append(dataset[feature][:])
        list_features.append(dataset["target"][:])
        data_tuple = np.asarray(tuple(zip(*list_features)))
        before = len(data_tuple)
        data_tuple = data_tuple[data_tuple[:,2]!=0] ##index 2 : trk1d0sig
        if (cat == "cat1"):
            print('Preselection (trk1d0sig !=0) applied, ', str(before - len(data_tuple)), ' jets did not pass cut')
        if (cat == "cat1"):  # nvtx==0
            data_tuple=data_tuple[data_tuple[:,0]==0]   ## index 0 : nvtx, index 1: nvtxall
        elif (cat == "cat2"): #nvtx==1&&nvtxall==1
            data_tuple=data_tuple[data_tuple[:,0]==1]
            data_tuple=data_tuple[data_tuple[:,1]==1]
        elif (cat == "cat3"): #nvtx==1&&nvtxall==2
            data_tuple=data_tuple[data_tuple[:,0]==1]
            data_tuple=data_tuple[data_tuple[:,1]==2]
        elif (cat == "cat4"): #nvtx>=2
            data_tuple=data_tuple[data_tuple[:,0]>=2]
        list_datasets.append(data_tuple[:,2:])
    
    dataset_all=np.concatenate((list_datasets[0],list_datasets[1],list_datasets[2]))
    data_tensor=torch.from_numpy(np.asarray(dataset_all))
    return data_tensor

# function for converting hdf5 file to torch tensor if no categorization is used, preselection is applied 
def ConvertToTensor_NoCat(data, features):
    list_datasets = []
    for dataset in data:
        list_features = []
        for feature in features:
            list_features.append(dataset[feature][:])
        list_features.append(dataset["target"][:])
        data_tuple = np.asarray(tuple(zip(*list_features)))
        before = len(data_tuple)
        data_tuple = data_tuple[data_tuple[:,0]!=0] ##index 0 : trk1d0sig
        print('Preselection (trk1d0sig !=0) applied, ', str(before - len(data_tuple)), ' jets did not pass cut')
        list_datasets.append(data_tuple)
    
    dataset_all=np.concatenate((list_datasets[0],list_datasets[1],list_datasets[2]))
    data_tensor=torch.from_numpy(np.asarray(dataset_all))
    return data_tensor

def Normalize(data_tensor, n_features):
    loader = torch.utils.data.DataLoader(data_tensor[:,0:n_features], batch_size=len(data_tensor), num_workers=1)
    data_for_norm = next(iter(loader))
    mean = torch.mean(data_for_norm, dim=0)
    std = torch.std(data_for_norm, dim=0)
    data_tensor[:,0:n_features] = (data_tensor[:,0:n_features]-mean)/std
    return data_tensor
    
def Splitting(data_tensor):
    n_train = int(len(data_tensor)*0.5)
    n_test = int(len(data_tensor)*0.25)
    n_val = int(len(data_tensor)*0.25)
    n_tot = n_train + n_test + n_val
    #prevent rounding errors:
    if n_tot < len(data_tensor):
        n_train = int(len(data_tensor)*0.5) + (len(data_tensor)- n_tot)
    elif n_tot > len(data_tensor):
        n_test = int(len(data_tensor)*0.25) - (n_tot - len(data_tensor))
    return n_train, n_test, n_val
        
def WeightClasses(train, n_features):
    loader = torch.utils.data.DataLoader(train, batch_size=len(train))
    data_for_weighting=next(iter(loader))
    class_sample_count = np.array( [len(np.where(data_for_weighting[:,n_features] == t)[0]) for t in range(3)])
    total=sum([class_sample_count[i] for i in range(3)])
    weight = total / class_sample_count /3
    samples_weight = np.array([weight[t] for t in range(3)])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.float()
    return samples_weight

def ConvertDataTypes(trainloader, n_features):
    inputs=[]
    labels=[]
    for i, data in enumerate(trainloader, 0):
            inp = data[:,0:n_features]
            inp=inp.float()
            inputs.append(inp)
            label = data[:,n_features]
            label=label.long()
            labels.append(label)
    return inputs, labels

def DoCalcForAccuracy(total, correct, class_total, class_correct, output, labels, misid_casb,misid_udsasb,misid_basc,misid_udsasc):
    
    _, predicted = torch.max(output, 1)
            
    total +=labels.size(0)
    correct += (predicted == labels).sum().item()
    misid_casb += (predicted[labels==1] == 0).sum().item()
    misid_udsasb += (predicted[labels==2] == 0).sum().item()
    misid_basc += (predicted[labels==0] == 1).sum().item()
    misid_udsasc += (predicted[labels==2] == 1).sum().item()
    if (len(labels)>1):
        c = (predicted == labels).squeeze()
    else:
        c = (predicted==labels)
    for j in range(len(labels)):
        label_item=labels[j]
        class_correct[label_item] += c[j].item()
        class_total[label_item] += 1
        #class_total an [1] --> number of c jets
    return total, correct, class_total, class_correct, misid_casb, misid_udsasb, misid_basc, misid_udsasc


# define parameters for plots
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',
          'axes.labelpad':'10'}

def PlotLearningCurves(training_loss, validation_loss, training_accuracy, validation_accuracy, training_accuracy_b, training_accuracy_c, training_accuracy_o,validation_accuracy_b,validation_accuracy_c,validation_accuracy_o, training_misidbasc, training_misidudsasc, training_misidudsasb, training_misidcasb, validation_misidbasc, validation_misidudsasc, validation_misidudsasb, validation_misidcasb, name):
    
    pylab.rcParams.update(params)
    
    fig=plt.figure(figsize=(10,20))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_yscale('log')
    names=[]
    plt.plot(training_loss)
    names.append('training')
    plt.plot(validation_loss)
    names.append('validation')
    plt.title('')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    plt.savefig('plots/'+name+'/loss.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    names=[]
    plt.plot(training_loss)
    names.append('training')
    plt.plot(validation_loss)
    names.append('validation')
    plt.title('')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim([0.5, 2.])
    plt.legend(names, loc='upper right')
    plt.savefig('plots/'+name+'/loss_smallerscale.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_accuracy)
    plt.plot(validation_accuracy)
    plt.title('')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='lower right')
    plt.savefig('plots/'+name+'/accuracy.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_accuracy_b)
    plt.plot(validation_accuracy_b)
    plt.title('b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='lower right')
    plt.savefig('plots/'+name+'/accuracy_b.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_accuracy_c)
    plt.plot(validation_accuracy_c)
    plt.title('c')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='lower right')
    plt.savefig('plots/'+name+'/accuracy_c.png')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(training_accuracy_o)
    plt.plot(validation_accuracy_o)
    plt.title('other')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='lower right')
    plt.savefig('plots/'+name+'/accuracy_o.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_misidbasc)
    plt.plot(validation_misidbasc)
    plt.title('b misidentified as c')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    plt.savefig('plots/'+name+'/misidbasc.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_misidudsasc)
    plt.plot(validation_misidudsasc)
    plt.title('uds misidentified as c')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    plt.savefig('plots/'+name+'/misidudsasc.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_misidcasb)
    plt.plot(validation_misidcasb)
    plt.title('c misidentified as b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    plt.savefig('plots/'+name+'/misidcasb.png')
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(training_misidudsasb)
    plt.plot(validation_misidudsasb)
    plt.title('uds misidentified as b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    plt.savefig('plots/'+name+'/misidudsasb.png')
    plt.close()
    
    
def PlotRoc(fpr1,tpr1,fpr2,tpr2,xlabel,leg1,leg2,name):
    pylab.rcParams.update(params)
    fig=plt.figure(figsize=(10,20))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_yscale('log')
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    plt.title('')
    label1= 'AUC = %0.2f' % roc_auc1
    label2= 'AUC = %0.2f' % roc_auc2
    plt.plot(tpr1, fpr1, 'g')
    plt.plot(tpr2, fpr2, 'b')
    leg1=leg1+', '+label1
    leg2=leg2+', '+label2
    names=[leg1,leg2]
    plt.legend(names, loc = 'lower right')
    plt.xlim([0.0001, 1])
    plt.ylim([0.0001, 1])
    plt.ylabel('Background rate')
    plt.xlabel(xlabel)
    plt.savefig(name)
    plt.close()
    
def PlotOutput(signal, bg1, bg2, leg1, leg2, leg3, name):
    pylab.rcParams.update(params)
    fig=plt.figure(figsize=(10,20))
    ax = fig.add_subplot(2, 1, 1)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    plt.hist(signal.numpy(),bins=50, color='r', histtype='step')
    plt.hist(bg1.numpy(),bins=50, color='b', histtype='step')
    plt.hist(bg2.numpy(),bins=50, color='g', histtype='step')
    names = [leg1, leg2, leg3]
    plt.legend(names, loc = 'upper right')
    plt.savefig(name)
    plt.close()
    
def PlotConfusion(matrix,name):
    pylab.rcParams.update(params)
    labels=['b jet','c jet','light jet']
    fig=plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2, 1, 1)
    for (j,i),label in np.ndenumerate(matrix):
        ax.text(i,j,("%.2f" % label ),ha='center',va='center')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    cb = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(cb, ax=ax)
    plt.savefig(name)
    plt.close()
    

def PlotPCA(X, name):
    cov_mat = np.cov(X.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vals_cor = np.corrcoef (eig_vals)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    #choosing the PCA for new feature space

    tot = sum(eig_vals)

    # explained variance
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(1)

        plt.bar(range(44), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(44), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('plots/'+name+'/PCA.png')    
    
    