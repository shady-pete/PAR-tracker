from net.MultitaskNet import MultiTaskNet, MultitaskCNNNet

if __name__ == '__main__':
    model1 = MultitaskCNNNet(train_backbone=False)
    model_gender = MultitaskCNNNet(train_backbone=True)
    model_hat = MultitaskCNNNet(train_backbone=True)
    model_bag = MultitaskCNNNet(train_backbone=True)

    model_gender.load_model('models/validation/_45.pth')
    model_hat.load_model('models_v2/validation/_65.pth')
    model_bag.load_model('models_v2/validation/_5.pth')

    model1.load_model('./models/model.pth')


    model1.gender_head = model_gender.gender_head
    model1.bag_head = model_bag.bag_head
    model1.hat_head = model_hat.hat_head

    model1.save_model("models/Finale3.pth")