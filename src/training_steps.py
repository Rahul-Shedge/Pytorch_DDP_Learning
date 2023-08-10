import torch
from tqdm.auto import tqdm
from typing import Dict,Tuple,List


def train_steps(
        model:torch.nn.Module,
        dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rank:int
    )-> Tuple[float,float]:

    """
    Args:

    """

    model.train()

    train_loss,train_acc = 0.0,0.0

    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(rank),y.to(rank)

        ypred = model(X)

        loss = loss_fn(ypred,y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class =  torch.argmax(torch.softmax(ypred,dim=1),dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(ypred)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss,train_acc


def test_steps(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              rank:int) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(rank), y.to(rank)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



def train(
    model:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    test_dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    epochs:int,
    # device:torch.device,
    rank_or_gpu_id:int
)-> Dict[str,List]:

    results = {
        "train_loss" : [],
        "train_acc" : [],
        "test_loss" : [],
        "test_acc" : []
    }


    model.to(rank_or_gpu_id)

    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_steps(
            model = model,
            dataloader= train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            rank=rank_or_gpu_id

        )

        test_loss,test_acc = test_steps(
            model = model,
            dataloader= test_dataloader,
            loss_fn=loss_fn,
            rank=rank_or_gpu_id

        )


            # Print out what's happening
        print(
          f"gpu_id : {rank} | "
          f"Batch_Size : {len(next(iter(train_dataloader))[0])} | "
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results





