import numpy as np


if __name__ == '__main__':
    namevec = np.load(file='facebase/name_of_fvec.npy')
    fvec = np.load(file='facebase/feature_vectors.npy')
    fnorm = np.load(file='facebase/norm_of_fvec.npy')
    print('Options:', '1) display names', '2) remove name', '3) save changes', '4) quit', sep='\n')

    while True:
        choise = input()
        if choise == 'display names' or choise == '1':
            ndict = {}
            for n in namevec:
                if n in ndict:
                    ndict[n] += 1
                else:
                    ndict[n] = 1
            for k, v in ndict.items():
                print('{} has {} photos'.format(k, v))
            print("Done")

        elif choise == 'remove name' or choise == '2':
            name = input('Enter the name you want to remove: ')
            found_indx = []
            for i, n in enumerate(namevec):
                if n == name:
                    found_indx.append(i)
            # remove them from fvec, namevec, fnorm
            namelst = [n for i, n in enumerate(namevec) if i not in found_indx]
            fveclst = [n for i, n in enumerate(namevec) if i not in found_indx]
            fnormlst = [n for i, n in enumerate(namevec) if i not in found_indx]
            namevec = np.array(namelst)
            fvec = np.array(fveclst)
            fnorm = np.array(fnorm)
            print('Done')

        elif choise == 'save changes' or choise == '3':
            print('Saving')
            if namevec.shape[0] == 0:
                namevec = np.array(['None'])
                fvec = np.ones((512, )) * 1000
                fnorm = np.ones((1,)) * 0.001
            np.save(file='facebase/name_of_fvec.npy', arr=namevec)
            np.save(file='facebase/feature_vectors.npy', arr=fvec)
            np.save(file='facebase/norm_of_fvec.npy', arr=fnorm)
            print('Done')
        elif choise == 'quit' or choise == '4' or choise == 'q':
            print('quitting')
            break

