import methode_pytorch as mp


def main():
    n_epoch = 30
    loss_dense, acc_dense = mp.model_dense(N_EPOCH=n_epoch)
    loss_cnn, acc_cnn = mp.model_cnn(N_EPOCH=n_epoch)
    mp.plot_result(n_epoch,loss_dense,loss_cnn,"Comparaison des loss")
    mp.plot_result(n_epoch,acc_dense,acc_cnn,"Comparaison des accuracy")


if __name__ == "__main__":
    main()