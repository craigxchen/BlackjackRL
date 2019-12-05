from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_v(V, usable_ace = False):
        fig, ax = plt.subplots()
        ax = Axes3D(fig)
        
        states = list(V.keys())
        states_YESace = {}
        states_NOTace = {}
        for state in states:
            if not state[2]:
                states_NOTace[state] = V[state]
            else: 
                states_YESace[state] = V[state]
        
        if usable_ace == 1:
            player_sum = [state[0] for state in states_YESace.keys()]
            dealer_show = [state[1] for state in states_YESace.keys()]
            scores = [val for val in states_YESace.values()]

            ax.plot_trisurf(player_sum, dealer_show, scores, cmap="viridis", edgecolor="none")
            ax.set_xlabel("Player's Sum")
            ax.set_ylabel("Dealer's Show Card")
            ax.set_zlabel("Perceived Value")
            ax.set_title("Soft Sums")
            ax.view_init(elev=40, azim=-100)
        else:
            player_sum = np.array([state[0] for state in states_NOTace.keys()])
            dealer_show = np.array([state[1] for state in states_NOTace.keys()])
            scores = np.array([val for val in states_NOTace.values()])

            ax.plot_trisurf(player_sum, dealer_show, scores, cmap="viridis", edgecolor="none")
            ax.set_xlabel("Player's Sum")
            ax.set_ylabel("Dealer's Show Card")
            ax.set_zlabel("Perceived Value")
            ax.set_title("Hard Sums")
            ax.view_init(elev=40, azim=-100)
        return


def plot_policy(policy, usable_ace = False):
    if not usable_ace:
        data = np.empty((18, 10))
        for state in policy.keys():
            if state[2] == usable_ace: 
                data[state[0]-4][state[1]-1] = policy[state]
    else:
        data = np.empty((10, 10))
        for state in policy.keys():
            if state[2] == usable_ace:
                data[state[0]-12][state[1]-1] = policy[state]
    
    # create discrete colormaps
    cmap = colors.ListedColormap(['red', 'green'])
    bounds = [-0.5,0.5,1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)
    
    ax.set_xticks(np.arange(10))
    if not usable_ace:
        ax.set_yticks(np.arange(18))
        ax.set_yticklabels(['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 
                            '14', '15', '16', '17', '18', '19', '20', '21'])
    else:
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels(['A + 1', 'A + 2', 'A + 3', 'A + 4', 'A + 5', 'A + 6', 
                            'A + 7', 'A + 8', 'A + 9', 'A + 10'])
        
    ax.set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    
    ax.set_xlabel('Dealer Show Card')
    if not usable_ace:
        ax.set_ylabel('Player Sum (Hard)')
    else:
        ax.set_ylabel('Player Sum (Soft)')
    
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.invert_yaxis()
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.4)
    # manually define a new patch 
    patch1 = mpatches.Patch(color='green', label='Hit')
    patch2 = mpatches.Patch(color='red', label='Stand')   
    # plot the legend
    plt.legend(handles=[patch1, patch2], loc='upper right')
    
    plt.show()
    return