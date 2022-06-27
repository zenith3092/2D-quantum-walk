# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

class TWO_D_RANDOM_WALK:
    def __init__(self, length):
        self.length = length
        self.size = length**2
        self.matrix = np.zeros((self.size, self.size))
        self.eigenvalues = None
        self.eigenvectors = None
        self.coefficients = None
        self.classical_prob = None
        self.quantum_prob = None
        self.xx = np.linspace(1, self.length, self.length)
        self.yy = np.linspace(1, self.length, self.length)
    
    def limited_boundary_matrix(self):
        for item in range(self.size):
            if (item % self.length == 0) and (item - self.length < 0):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 2
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif (item % self.length == self.length-1) and (item - self.length < 0):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 2
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif (item % self.length == 0) and (item + self.length + 1 > self.length**2):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 2
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
            elif (item % self.length == self.length - 1) and (item + self.length + 1 > self.length**2):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 2
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
            elif item % self.length == 0:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 3
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif item % self.length == self.length - 1:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 3
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif item - self.length < 0:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 3
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif item + self.length + 1 > self.length**2:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 3
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
            else:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1

    def periodic_boundary_matrix(self):
        for item in range(self.size):
            if (item % self.length == 0) and (item - self.length < 0):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item + (self.length - 1): 
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length * (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif (item % self.length == self.length - 1) and (item - self.length < 0):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length * (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif (item % self.length == 0) and (item + self.length + 1 > self.length**2):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item + (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length * (self.length - 1):
                        self.matrix[item][juice] = -1
            elif (item % self.length == self.length - 1) and (item + self.length + 1 > self.length**2):
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length * (self.length - 1):
                        self.matrix[item][juice] = -1
            elif item % self.length == 0:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item + (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif item % self.length == self.length - 1:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif item - self.length < 0:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length * (self.length - 1):
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
            elif item + self.length + 1 > self.length**2:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length * (self.length - 1):
                        self.matrix[item][juice] = -1
            else:
                for juice in range(self.size):
                    if item == juice :
                        self.matrix[item][juice] = 4
                    elif juice == item - 1:
                        self.matrix[item][juice] = -1
                    elif juice == item + 1:
                        self.matrix[item][juice] = -1
                    elif juice == item - self.length:
                        self.matrix[item][juice] = -1
                    elif juice == item + self.length:
                        self.matrix[item][juice] = -1
    
    def generate_eig(self):
        eigenvalues, eigenvectors = la.eig(self.matrix)
        eigenvalues = np.array([eigenvalues])
        eigenvalues = eigenvalues.T
        eigenvectors_T = eigenvectors.T
        eigenvectors_T = np.real(eigenvectors_T)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors_T
    
    def generate_coefficients(self, p_0):
        _coe = np.zeros((self.size, 1))
        for item in range(self.size):
            _coe[item] = np.array([self.eigenvectors[item]]) @ p_0
        self.coefficients = _coe

    def classical_p(self, t):
        prob = np.zeros((1, self.size))
        for item in range(self.size):
            prob = prob + self.coefficients[item][0] * np.exp(-1 * self.eigenvalues[item][0] * t) * self.eigenvectors[item]
        prob_T = prob.T
        c_p = prob_T.reshape((self.length, self.length))
        c_p = np.real(c_p)
        self.classical_prob = c_p
        return c_p
    
    def quantum_p(self, t):
        prob_amplitude = np.zeros((1, self.size))
        prob = np.zeros((1, self.size))
        for item in range(self.size):
            prob_amplitude = prob_amplitude + self.coefficients[item][0] * np.exp(-1j * self.eigenvalues[item][0] * t) * self.eigenvectors[item]
        for item in range(self.size):
            prob[0][item]= (abs(prob_amplitude[0][item]))**2
        prob_T = prob.T
        q_p = prob_T.reshape((self.length, self.length))
        self.quantum_prob = q_p
        return q_p
    
    def export_one_image(self, output_time, save=False, filename=None, savetype=None):
        X, Y = np.meshgrid(self.xx, self.yy)

        c_prob = self.classical_p(output_time)
        q_prob = self.quantum_p(output_time)

        plt.figure(figsize=(25, 40))
        plt.rcParams['axes.facecolor'] = "gray"

        ax = plt.subplot(1, 2, 1, projection='3d')
        ax.contourf(X, Y, c_prob, zdir='z', offset=-2, cmap='terrain')
        surface = ax.plot_surface(X, Y, c_prob, rstride=1, cstride=1, cmap='terrain', edgecolor='none')
        ax.set_title("Classical Walk", fontsize=70)
        ax.set_xlabel('x', fontsize=30)
        ax.set_ylabel('y', fontsize=30)
        ax.set_zlabel('p ', fontsize=30)
        ax.set_zlim(0, 0.05)
        ax.text2D(0.05, 0.05, "t="+str(output_time), fontsize=50, transform=ax.transAxes)

        ax2 = plt.subplot(1,2,2,projection='3d')
        ax2.contourf(X, Y, q_prob, zdir='z', offset=-2, cmap='terrain')
        surface2 = ax2.plot_surface(X, Y, q_prob, rstride=1, cstride=1, cmap='terrain', edgecolor='none')
        ax2.set_title("Quantum Walk", fontsize=70)
        ax2.set_xlabel('x', fontsize=30)
        ax2.set_ylabel('y', fontsize=30)
        ax2.set_zlabel('p ', fontsize=30)
        ax2.text2D(1.125, 0.05, "t="+str(output_time), fontsize=50, transform=ax.transAxes)
        ax2.set_zlim(0, 0.05)
        plt.tight_layout()
        self.save_image(save, inputtime=output_time, filename=filename, savetype=savetype)

    def export_several_image(self, start_time, end_time, save=False, filename=None, savetype=None):
        for time in range(start_time, end_time):
            X, Y = np.meshgrid(self.xx, self.yy)
            
            c_prob = self.classical_p(time)
            q_prob = self.quantum_p(time)
            
            plt.figure(figsize=(25, 40))
            plt.rcParams['axes.facecolor'] = "gray"
            
            ax = plt.subplot(1, 2, 1, projection='3d')
            ax.contourf(X, Y, c_prob, zdir='z', offset=-2, cmap='terrain')
            surface = ax.plot_surface(X, Y, c_prob, rstride=1, cstride=1, cmap='terrain', edgecolor='none')
            ax.set_title("Classical Walk", fontsize=70)
            ax.set_xlabel('x', fontsize=30)
            ax.set_ylabel('y', fontsize=30)
            ax.set_zlabel('p ', fontsize=30)
            ax.set_zlim(0, 0.05)
            ax.text2D(0.05, 0.05, "t="+str(time/10),fontsize=50, transform=ax.transAxes)
            
            ax2 = plt.subplot(1, 2, 2, projection='3d')
            ax2.contourf(X, Y, q_prob, zdir='z', offset=-2, cmap='terrain')
            surface2 = ax2.plot_surface(X, Y, q_prob, rstride=1, cstride=1, cmap='terrain', edgecolor='none')
            ax2.set_title("Quantum Walk", fontsize=70)
            ax2.set_xlabel('x', fontsize=30)
            ax2.set_ylabel('y', fontsize=30)
            ax2.set_zlabel('p ', fontsize=30)
            ax2.text2D(1.125, 0.05, "t=" + str(time/10), fontsize=50, transform=ax.transAxes)
            ax2.set_zlim(0, 0.05)
            plt.tight_layout()
            self.save_image(save, inputtime=time, filename=filename, savetype=savetype)
    
    def save_image(self, save, inputtime, filename, savetype):
        if save:
            if inputtime:
                if filename:
                    if savetype:
                        plt.savefig(filename + str(inputtime) + '.' + savetype, bbox_inches='tight', pad_inches=0, format=savetype)
                    else:
                        plt.savefig(filename + str(inputtime) + '.jpg', bbox_inches='tight', pad_inches=0, format='jpg')
                else:
                    if savetype:
                        plt.savefig('output' + str(inputtime) + '.' + savetype, bbox_inches='tight', pad_inches=0, format=savetype)
                    else:
                        plt.savefig('output' + str(inputtime) + '.jpg', bbox_inches='tight', pad_inches=0, format='jpg')
            else:
                if filename:
                    if savetype:
                        plt.savefig(filename, bbox_inches='tight', pad_inches=0, format=savetype)
                    else:
                        plt.savefig(filename + '.jpg', bbox_inches='tight', pad_inches=0, format='jpg')
                else:
                    if savetype:
                        plt.savefig('output' + '.' + savetype, bbox_inches='tight', pad_inches=0, format=savetype)
                    else:
                        plt.savefig('output' + '.jpg', bbox_inches='tight', pad_inches=0, format='jpg')

if __name__ == '__main__':
    ## define length by length lattice 
    length = 64
    size = length**2
    
    ## define initial state
    p_0 = np.zeros((size, 1))
    p_0[0] = 1.

    ## add an object
    test = TWO_D_RANDOM_WALK(length)
    
    ## define a matrix with boundary conditions
    test.limited_boundary_matrix()
    # test.periodic_boundary_matrix()
    print(test.matrix)

    ## calculate eigenvalues and eigenvectors
    test.generate_eig()
    print(test.eigenvalues)
    print(test.eigenvectors)

    ## calculate coefficients
    test.generate_coefficients(p_0)
    print(test.coefficients)

    ## calculate probability in classical physics
    test.classical_p(15)  
    print(test.classical_prob)

    ## calculate probability in classical physics
    test.quantum_p(15)  
    print(test.quantum_prob)

    ## export an image
    test.export_one_image(output_time=15, save=True, filename='hola', savetype='png')

    ## export several images
    # test.export_several_image(start_time=1, end_time=15, save=True, filename='report', savetype='jpg')