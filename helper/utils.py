def train(data_loader, net, optimizer, loss_graph):
    
    for i, data in enumerate(data_loader):
        
        inputs, masks = data
        if USE_GPU:
            inputs = inputs.cuda()
            masks = masks.cuda()
            net = net.cuda()
        
        # Write me
        optimizer.zero_grad()  # zero out the existing gradients
            
        main_loss = net(inputs, gts = masks)
        loss_graph.append(main_loss.item())
        main_loss.backward() # compute the gradient and add to computation tree

        optimizer.step()  # update the parameters according to the gradients
    return main_loss

def train_plot(epoch, data_loader, net, optimizer, loss_graph, detail=False):
    net.train()
    print("Starting Training...")
    loss_graph = []

    fig = plt.figure(figsize=(12,6))
    plt.subplots_adjust(bottom=0.2,right=0.85,top=0.95)
    ax = fig.add_subplot(1,1,1)
    for e in range(epoch):
        loss = train(data_loader, net, optimizer, loss_graph)
        ax.clear()
        ax.set_xlabel('iterations')
        ax.set_ylabel('loss value')
        ax.set_title('Training loss curve for OVERFIT_NET')
        ax.plot(loss_graph, label='training loss')
        ax.legend(loc='upper right')
        fig.canvas.draw()
        if detail== False and e%10==0:
            print("Epoch: {} Loss: {}".format(e, loss))
        elif detail== True:
            print("Epoch: {} Loss: {}".format(e, loss))
    return net

def validate(val_loader, net):
    
    iou_arr = []
    val_loss = 0
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
        
            inputs, masks = data

            if USE_GPU:
                inputs = inputs.cuda()
                masks = masks.cuda()
                net = net.cuda()

            # Hint: make sure the range of values of the ground truth is what you expect
            output = net(inputs)
            val_loss += nn.CrossEntropyLoss(ignore_index=255)(output, masks)
            _, indices = torch.max( output, dim=1 )
            preds = indices.cpu().numpy()[0][None]
            gts = torch.from_numpy(np.array(masks.cpu(), dtype = np.int32)).long().numpy()
            gts[gts == 255] = -1
            
            conf = eval_semantic_segmentation(preds, gts)

            iou_arr.append(conf['miou'])
    
    return val_loss, (sum(iou_arr) / len(iou_arr))

def getmIoU(net, tf, img, target):
    
    transformedImg, transformedTarget = tf(img, target)
    if USE_GPU:
        net = net.cuda()
        transformedImg = transformedImg.cuda()
        transformedTarget = transformedTarget.cuda()
        
    output = net.forward(transformedImg[None])
    if USE_GPU:
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    else:
        pred = torch.argmax(output, dim=1).numpy()[0]
    truth = torch.from_numpy(np.array(target.convert('P'), dtype=np.int32)).long().numpy()
    truth[truth==255] = -1
    res = eval_semantic_segmentation(pred[None], truth[None])
    return transformedImg, output, res["miou"]

def plotPrediction(net, tf, img, target):
    net.eval()
    
    transformedImg, output, miou = getmIoU(net, tf, img, target)
    
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1,3,1)
    plt.title('Original Image')
    ax.imshow(img)
    ax = fig.add_subplot(1,3,2)
    plt.title('Target')
    ax.imshow(target)
    ax = fig.add_subplot(1,3,3)
    plt.title('Prediction')
    ax.text(10, 25, 'mIoU = {:_>8.6f}'.format(miou), fontsize=15, color='white')
    ax.imshow(colorize_mask(torch.argmax(output, dim=1).cpu().numpy()[0]))
    

    
    
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
