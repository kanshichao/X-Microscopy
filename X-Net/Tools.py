import  pickle as pkl

def create_loss_G():
    loss_G = {'errGa':[0],'errGb':[0],'errGf':[0]}
    fw = open('loss_G','wb')
    pkl.dump(loss_G,fw)

def load_loss_G():
    # loss_G = {'errGa':[],'errGb':[],'errGf':[]}
    rw = open('loss_G','rb')
    a = pkl.load(rw)
    print(a)

if __name__ == '__main__':
    create_loss_G()
    load_loss_G()