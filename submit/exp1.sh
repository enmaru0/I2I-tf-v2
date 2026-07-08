EXP_DIR=results/exp_0001

# 1mmで臓器周辺のみを学習させるためのスクリプト
OPTIONS="--overrides
        exp_dir=${EXP_DIR}
        aug.random_crop_method.body=0.0
        aug.random_crop_method.organ=0.9
        aug.random_crop_method.organ_crop=0.1
        aug.random_crop_method.image=0.0
        aug.affine.norm_spacing_zyx=[1.0,1.0,1.0]
        data_dir=datasets_prostate
        model.renorm.r_max=1.0 model.renorm.d_max=0.0 
        "

echo ${OPTIONS}
# 2バッチ分の画像をデバッグ用に保存する
python debug_dataloader.py ${OPTIONS}
python main.py ${OPTIONS}
python predict.py ${EXP_DIR}/checkpoints/model_latest.keras 
python utils/convert2binary.py ${EXP_DIR}/preds
python utils/calculate_dice.py ${EXP_DIR}/preds datasets_prostate/val
python export_params.py ${EXP_DIR}/checkpoints/model_latest.keras --param_name CNNHigh
