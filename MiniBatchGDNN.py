import numpy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy




class MiniBatchGDNN:
    
    def __init__(self, k, d):
        # k is the number of classes
        # d is the data dimention
        self.k = k
        self.d = d

        # initializing the weight matrix
        self.W = np.random.normal(0, 0.01, (k, d))
        self.b = np.random.normal(0, 0.01, (k, 1))

    
    def evaluate_classifier(self, X, W, b):
        # Each column in X corresponds to 'one' image in this context having the d*n size
        # This function returns: k*n, each row contains probability for the specific class

        if len(X.shape) >= 2:
            second_dimention = X.shape[1]
            temp_b = np.array(b * second_dimention)
            result = np.dot(W, X) + temp_b
        else:
            second_dimention = 0
            temp_b = b
            result = np.dot(W, X).reshape(10, 1) + temp_b
        
        

        P = np.zeros(result.shape).shape
        
        if second_dimention:
            P = np.zeros(result.shape)
            for j in range(second_dimention):
                P[:, j] = np.exp(result[:, j]) / sum(np.exp(result[:, j]))
        else:
            P = np.zeros((result.shape[0], 1))
            P = (np.exp(result) / sum(np.exp(result))).reshape((self.k, 1))
            # just one row of elements in P
        
        return P
    

    def compute_cost(self, X, Y, W, b, lambda_):
        n = X.shape[1]
        r = np.sum(np.square(self.W))
        P = self.evaluate_classifier(X, W, b)
        l = 0
        for i in range(n):
            y = Y[:, i]
            p = P[:, i]
            probability = np.dot(y, p)
            
            if probability == 1:
                print('!!!!!!!!!!!!!!!!!!!+++++++++++++!!!!!!!!!!!!!!!!!!!!')
                print('probability is 1 => y: \n{0}\np:\n{1}'.format(y, p))

            if probability == 0:
                print('!!!!!!!!!!!!!!!!!!!_______!!!!!!!!!!!!!!!!!!!!')
                print('probability is 0 => y is {0}\n p is: {1}'.format(y, p))
                print('!!!!!!!!!!!!!!!!!!!_______!!!!!!!!!!!!!!!!!!!!')
                
                probability = 1.00e-100

            
            l = - np.log(probability)
            

            if not l:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('"not l" was true, l is: {0}, probablity: {1}'.format(l, probability))
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
            assert not np.isinf(l)
            
        J = (l / n) + (lambda_ * r)
        
        return J
    
    def accuracy(self, X, y, W, b):
        # y is a vector containing the ground truth label numbers (just like train_y)
        P = self.evaluate_classifier(X, W, b)
        prediction = np.argmax(P, axis=0)
        n = X.shape[1]
        incorrect = 0
        for i in range(n):
            if y[i] != prediction[i]:
                incorrect += 1
        return (n - incorrect) / n
    

    def compute_grads_num_slow(self, X, Y, P, W, b, lamda_, h=0.01):
        

        no 	= 	W.shape[0]
        d 	= 	X.shape[0]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros((no, 1))


        
        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] -= h
            c1 = self.compute_cost(X, Y, W, b_try, lamda_)

            b_try = np.array(b)
            b_try[i] += h
            c2 = self.compute_cost(X, Y, W, b_try, lamda_)

            grad_b[i] = (c2-c1) / (2*h)
        

        for i in range(W.shape[0]):
            print('>> inside grad computation main loop, i: {}'.format(i))
            for j in range(W.shape[1]):

                W_try = np.array(W)
                W_try[i,j] -= h
                c1 = self.compute_cost(X, Y, W_try, b, lamda_)

                W_try = np.array(W)
                W_try[i,j] += h
                c2 = self.compute_cost(X, Y, W_try, b, lamda_)

                grad_W[i,j] = (c2-c1) / (2*h)

                
                if np.isnan(grad_W[i,j]):
                    print('You are in a bad situation...')
                    print(2*h)
                    print(c1)
                    print(c2)
                    print('X: \n {}'.format(X))
                    print('Y: \n {}'.format(Y))
                    print('Is W_try nan: {0}\nW_try: \n {1}'.format(np.isnan(np.sum(W_try)), W_try))
                    print('Is b nan: {0}\nb: \n {1}'.format(np.isnan(np.sum(b)), b))
                    print('lambda: {}'.format(lamda_))
                    print('#####################')
                
                

        if np.isnan(np.sum(grad_W)):
            print('\n.   ^^^^^^^^^^^^^^^^^^^^ smt bad: ')
        
        assert not np.isnan(np.sum(grad_W))

        return [grad_W, grad_b]



    def train(self, X, Y, eta=0.01, lambda_=0, epochs=100, mini_batch_size=100):
        # forward pass
        # backward pass
        # update weight matrix
        
        # let's do a permutation of the total dataset size!
        n = X.shape[1]
        d = X.shape[0]
        assert n % mini_batch_size == 0

        stacked_arrays = np.concatenate((X, Y))
        '''
        W = copy.deepcopy(self.W)
        temp_ar = np.zeros((self.k, n-d))
        temp_ar2 = np.concatenate((W, temp_ar),  axis=1)
        stacked_arrays = np.concatenate((stacked_arrays, temp_ar2))
        '''
        
        np.take(stacked_arrays, np.random.permutation(n), axis=1, out=stacked_arrays)
        
        X_, Y_ = np.split(stacked_arrays, [self.d], axis=0)
        
        '''
        Y_, W = np.split(Y_W, [self.k], axis=0)
        W = copy.deepcopy(W[:, 0:self.d])
        '''
        

        for i in range(epochs):
            '''
            # if we use train_y as well:
            temp1 = np.concatenate((train_X, train_Y))
            temp2 = train_y.reshape((train_y.shape[0], 1)).T
            stacked_arrays = np.concatenate((temp1, temp2)).shape
            '''
            

            # remember that you need to stack the newly changed W matrix on this not the old one!!!

            

            num_of_batches = int(n / mini_batch_size)
            l_index = 0
            h_index = 0
            for i in range(num_of_batches):
                l_index = h_index
                h_index = (i+1) * mini_batch_size
                batch_X = X_[:, l_index:h_index]
                batch_Y = Y_[:, l_index:h_index]
                P = self.evaluate_classifier(batch_X, self.W, self.b)

                grad_W, grad_b = self.compute_grads_num_slow(batch_X, batch_Y, P, self.W, self.b, lambda_)
                
                print('******** grads computed, batch number: {}'.format(i))
                
                # TODO: remove this:
                '''
                print('shape of W: {}'.format(self.W.shape))
                print('shape of grad_W: {}'.format(grad_W.shape))
                print('shape of b: {}'.format(self.b.shape))
                print('shape of grad_b: {}'.format(grad_b.shape))
                print('this is grad_W:')
                print(grad_W)
                print('dtype of grad_W: {0}, dtype of W: {1}'.format(self.W.dtype, grad_W.dtype))
                print('eta is: {}'.format(eta))
                print('result of operation (temp): ')
                print(self.W - (eta * grad_W))
                '''

                print('dimentions of grad_W: {}'.format(grad_W.shape))

                self.W = self.W - (eta * grad_W)
                self.b = self.b - (eta * grad_b)

                # TODO: remove this
                print('------------------------')
            

            
            stacked_arrays = np.concatenate((X_, Y_))
            '''
            temp_ar2 = np.concatenate((W, temp_ar),  axis=1)
            stacked_arrays = np.concatenate((stacked_arrays, temp_ar2))
            '''
            
            np.take(stacked_arrays, np.random.permutation(n), axis=1, out=stacked_arrays)
            
            X_, Y_ = np.split(stacked_arrays, [self.d], axis=0)
            '''
            Y_, W = np.split(Y_W, [self.k], axis=0)
            W = copy.deepcopy(W[:, 0:self.d])
            '''
            
            # TODO: remove this
            print('epoch ended...')

        
        # When all of epochs end, we have to replace the self.W with the heavily updated W!
        

        

