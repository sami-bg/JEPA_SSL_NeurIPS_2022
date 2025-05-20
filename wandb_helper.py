import wandb, datetime as dt
import pandas as pd
WANDB_ENTITY = 'sami-bg'
WANDB_PROJECT = 'HJEPA-debug'



api   = wandb.Api(timeout=45)
since = '2025-04-21T00:00:00Z'          # ISO-8601 strings work best
filters = {
    'createdAt'                     : {'$gte': since},
    'state'                         : 'finished',
    'config.dataset_structured_noise': False,
    'config.dataset_noise'          : 0.0,
    'config.vjepa.masking_ratio'    : 0.9,
    #  <-- no "group" key here -->
}

runs = api.runs(f"{WANDB_PROJECT}", filters=filters)

# ----- set best_performance_finetune_enc_100 -----
if False:
    runs_by_group = {}
    for run in runs:
        key = (run.config['vjepa']['tubelet_size'],
            run.config['dataset_static_noise'])
        runs_by_group.setdefault(key, []).append(run)

    for (tubelet, static_noise), group_runs in runs_by_group.items():
        best = min(
            (r for r in group_runs
            if not r.history(keys=['finetune_enc_100/loss']).empty),
            key=lambda r: r.history(keys=['finetune_enc_100/loss'])
                        ['finetune_enc_100/loss'].iloc[-1],
            default=None,
        )
        if best:
            best_loss = float(best.history(keys=['finetune_enc_100/loss'])\
                        ['finetune_enc_100/loss'].iloc[-1])
            best.summary.update({'best_performance_finetune_enc_100': best_loss})
            best.update()
            print(f"{tubelet=} {static_noise=} → {best.name}  loss={best_loss:.4f}")


csv_path = 'best_performance_finetune_enc_100.csv'
# ----- read best_performance_finetune_enc_100 and write into csv with columns:
# dataset_structured_noise, best_performance_finetune_enc_100, tubelet_size, dataset_static_noise, dataset_noise, masking_ratio, model_type (aka vjepa or hjepa)
# -----

# Create CSV with specified columns
data = []
for run in runs:
    if 'best_performance_finetune_enc_100' in run.summary:
        data.append({
            'dataset_structured_noise': run.config.get('dataset_structured_noise', False),
            'best_performance_finetune_enc_100': run.summary['best_performance_finetune_enc_100'],
            'tubelet_size': run.config['vjepa']['tubelet_size'],
            'dataset_static_noise': run.config.get('dataset_static_noise', 0.0),
            'dataset_noise': run.config.get('dataset_noise', 0.0),
            'masking_ratio': run.config['vjepa']['masking_ratio'],
            'model_type': 'vjepa' if 'vjepa' in run.config else 'hjepa'
        })

df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)
print(f"CSV file saved to {csv_path}")
# add this to a wandb custom line chart

# Create and log the custom charts to wandb
wandb.init(project=WANDB_PROJECT, name="performance_analysis")

# Group the data by model_type and tubelet_size
grouped_data = df.groupby(['model_type', 'tubelet_size'])

# Create chart for dataset_static_noise
static_noise_xs = []
static_noise_ys = []
static_noise_keys = []

for (model, tubelet), group in grouped_data:
    # Sort by noise level to ensure connected lines
    group_sorted = group.sort_values('dataset_static_noise')
    static_noise_xs.append(group_sorted['dataset_static_noise'].tolist())
    static_noise_ys.append(group_sorted['best_performance_finetune_enc_100'].tolist())
    static_noise_keys.append(f"{model}_tubelet{tubelet}")

wandb.log({
    "Static Noise vs Performance": wandb.plot.line_series(
        xs=static_noise_xs,
        ys=static_noise_ys,
        keys=static_noise_keys,
        title="Static Noise vs Best Performance",
        xname="Static Noise Level"
    )
})

# Create chart for dataset_noise
dynamic_noise_xs = []
dynamic_noise_ys = []
dynamic_noise_keys = []

for (model, tubelet), group in grouped_data:
    # Sort by noise level to ensure connected lines
    group_sorted = group.sort_values('dataset_noise')
    dynamic_noise_xs.append(group_sorted['dataset_noise'].tolist())
    dynamic_noise_ys.append(group_sorted['best_performance_finetune_enc_100'].tolist())
    dynamic_noise_keys.append(f"{model}_tubelet{tubelet}")

wandb.log({
    "Dynamic Noise vs Performance": wandb.plot.line_series(
        xs=dynamic_noise_xs,
        ys=dynamic_noise_ys,
        keys=dynamic_noise_keys,
        title="Dynamic Noise vs Best Performance",
        xname="Dynamic Noise Level"
    )
})

wandb.finish()
