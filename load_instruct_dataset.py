from datasets import load_dataset

dataset = load_dataset('databricks/databricks-dolly-15k', split='train')
print(len(dataset))
dataset = dataset.filter(lambda example: len(example["response"]) >= 1024 and example["response"])
print(len(dataset))

item = next(iter(dataset))
print(item['instruction'])
print(len(item['response']))


device = next(model.parameters()).device
cached_hiddens = []
for i in trange(len(dataloader), total=len(dataloader), desc="Caching hiddens", leave=False):
    # with torch.autocast(device_type="cuda", enabled=args.amp):
    batch = maybe_get_0th_element(dataloader[i]).to(device)
    cached_hiddens.append(model.model(batch).last_hidden_state.cpu())
return cached_hiddens