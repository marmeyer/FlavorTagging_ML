import torch
import torch.nn as nn
import torch.nn.functional as F
import helperfunctions
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics


def train(net, inputs, labels, n_epochs, criterion, optimizer, validationloader, n_features, model ):

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        for p in net.parameters():  # reset requires_grad
            p.requires_grad_(True)
        
        # initialize variables needed for learning curves
        running_loss = 0.0
        
        correct_training = 0
        total_training = 0
        class_correct_training = list(0. for i in range(3))
        class_total_training = list(0. for i in range(3))
        misid_casb_training = 0
        misid_udsasb_training = 0
        misid_basc_training = 0
        misid_udsasc_training = 0
        
        correct_validation = 0
        total_validation = 0
        class_correct_validation = list(0. for i in range(3))
        class_total_validation = list(0. for i in range(3))
        misid_casb_validation = 0
        misid_udsasb_validation = 0
        misid_basc_validation = 0
        misid_udsasc_validation = 0
        
             
        for i, data in enumerate(inputs, 0):
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, labels[i])
            loss.backward()
            optimizer.step()
            
            # determine the accuracy
            total_training, correct_training, class_total_training, class_correct_training, misid_casb_training,misid_udsasb_training,misid_basc_training,misid_udsasc_training  = helperfunctions.DoCalcForAccuracy(total_training, correct_training, class_total_training, class_correct_training, outputs.data, labels[i], misid_casb_training,misid_udsasb_training,misid_basc_training,misid_udsasc_training)
       
            running_loss += loss.item()
        
        #print and save trainings loss per epoch    
        print('[%d] training loss: %.6f' % (epoch, running_loss / len(inputs)))
        print('[%d] training accuracy %.6f: ' % (epoch, 100 * correct_training / total_training))
        for i in range(3):
            print('[%d] training accuracy of %5s : %2d %%' % (epoch, i, 100 * class_correct_training[i] / class_total_training[i]))

        #print and save validation loss and accuracy per epoch 
        with torch.no_grad():
            running_loss_val = 0.0
            for valdata in validationloader:    # rewrite such that function can be used
                inputs_val = valdata[:,0:n_features]
                inputs_val=inputs_val.float()
                labels_val = valdata[:,n_features]
                labels_val=labels_val.long()
                
                outputs_val=net(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                running_loss_val+=loss_val.item()
                
                total_validation, correct_validation, class_total_validation, class_correct_validation, misid_casb_validation, misid_udsasb_validation, misid_basc_validation, misid_udsasc_validation = helperfunctions.DoCalcForAccuracy(total_validation, correct_validation, class_total_validation, class_correct_validation, outputs_val.data, labels_val, misid_casb_validation, misid_udsasb_validation, misid_basc_validation, misid_udsasc_validation )
                
            print('[%d] validation loss: %.6f' % (epoch, running_loss_val / len(validationloader)))
            print('[%d] validation accuracy %.6f: ' % (epoch, 100 * correct_validation / total_validation))
            for i in range(3):
                print('[%d] validation accuracy of %5s : %2d %%' % (epoch, i, 100 * class_correct_validation[i] / class_total_validation[i]))

        # save checkpoints        
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': (running_loss / len(inputs)),
            'accuracy' : (100 * correct_training / total_training),
            'accuracy_b' : (100 * class_correct_training[0] / class_total_training[0]),
            'accuracy_c' : (100 * class_correct_training[1] / class_total_training[1]),
            'accuracy_o' : (100 * class_correct_training[2] / class_total_training[2]),
            'loss_val'   : (running_loss_val / len(validationloader)),
            'accuracy_val' : (100 * correct_validation / total_validation),
            'accuracy_b_val' : (100 * class_correct_validation[0] / class_total_validation[0]),
            'accuracy_c_val' : (100 * class_correct_validation[1] / class_total_validation[1]),
            'accuracy_o_val' : (100 * class_correct_validation[2] / class_total_validation[2]),
            'midis_b_as_c' : (100 * misid_basc_training / class_total_training[0]), 
            'midis_uds_as_c' : (100 * misid_udsasc_training / class_total_training[2]),
            'midis_c_as_b' : (100 * misid_casb_training / class_total_training[1]), 
            'midis_uds_as_b' : (100 * misid_udsasb_training / class_total_training[2]),
            'midis_b_as_c_val' : (100 * misid_basc_validation / class_total_validation[0]), 
            'midis_uds_as_c_val' : (100 * misid_udsasc_validation / class_total_validation[2]),
            'midis_c_as_b_val' : (100 * misid_casb_validation / class_total_validation[1]), 
            'midis_uds_as_b_val' : (100 * misid_udsasb_validation / class_total_validation[2])
        }, '/beegfs/desy/user/mameyer/checkpoints/'+ model +'_'+ str(epoch)+'.pth')    

    print('Finished Training')
    
    # save model
    torch.save(net.state_dict(), 'models/'+ model + '.pth')
