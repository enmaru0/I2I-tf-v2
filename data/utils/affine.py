import numpy as np
from scipy.ndimage import affine_transform

from .affine_utils import (
    create_scaling_matrix,
    create_translation_matrix,
    random_flipping,
    random_rotation,
    random_scaling,
    random_translation,
)
from .data_utils import eight_point_extractor, six_point_extractor, two_point_extractor


class AffineTransform:
    def __init__(
        self,
        crop_size_zyx: tuple[int, int, int],
        norm_spacing_zyx: tuple[float, float, float],
        random_rot_deg_zyx: tuple[int, int, int],
        random_flip_axis_zyx: tuple[bool, bool, bool],
        random_scaling_range_zyx: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        random_shift_rate_zyx: tuple[float, float, float],
    ):
        """
        3Dデータに対するアフィン変換クラス。

        このクラスは、スケーリング、回転、平行移動、反転のアフィン変換を取り扱います。
        トレーニング中にランダムな変換によるデータ拡張をサポートします。

        パラメータ
        ----------
        crop_size_zyx : tuple of int
            z, y, xの各次元でのクロップサイズを指定します。
        norm_spacing_zyx : tuple of float
            z, y, xの各次元での正規化された間隔を指定します。
        random_rot_deg_zyx : tuple of int
            z, y, xの各次元でのランダム回転角度を指定します。
            0-180の間の数字を指定してください。
        random_flip_axis_zyx : tuple of bool
            z, y, xの各次元でランダムに反転する軸を指定します。
        random_scaling_range_zyx : tuple of float
            ランダムスケーリングの範囲を指定します。
        random_shift_rate_zyx : tuple of float
            z, y, xの各次元でのランダムシフト率を指定します。
            画像の回転中心で制御できるのであまり使わないです。
            使わない場合は(0,0,0)と指定してください。
        """
        super().__init__()
        self.crop_size_zyx = np.array(crop_size_zyx)
        self.norm_spacing_zyx = np.array(norm_spacing_zyx)
        self.random_rot_deg_zyx = np.array(random_rot_deg_zyx)
        self.random_shift_rate_zyx = np.array(random_shift_rate_zyx)
        self.random_flip_axis_zyx = np.array(random_flip_axis_zyx)
        self.scaling_range_zyx = np.array(random_scaling_range_zyx)
        self._validate_parameters()

    def _validate_parameters(self):
        for i in range(3):
            assert 0 < self.norm_spacing_zyx[i], self.norm_spacing_zyx
            assert 0 <= self.random_rot_deg_zyx[i] <= 180, self.random_rot_deg_zyx
            assert 0 <= self.random_shift_rate_zyx[i] <= 1, self.random_shift_rate_zyx

            for j in range(2):
                assert 0 <= self.scaling_range_zyx[i][j] <= 1, self.scaling_range_zyx[
                    i
                ][j]

    def get_affine(
        self,
        img_spacing_zyx: tuple[float, float, float],
        crop_center_zyx: tuple[float, float, float],
        is_training: bool,
        return_random_var: bool = False,
        random_var: dict = {},
        rng=None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """
        パラメータに基づいてアフィン変換行列を生成します。

        パラメータ
        ----------
        img_spacing_zyx : tuple of float
            入力画像のz, y, x方向の間隔を指定します。
        crop_center_zyx : tuple of float
            クロップの中心座標をz, y, xの順で指定します。
        is_training : bool
            トレーニング中かどうかを指定します (ランダム変換を適用するかどうか)。
        return_random_var : bool, optional
            Trueの場合、変換に使用されたランダム変数を返します。
        random_var : dict, optional
            事前に計算されたランダム変数を含む辞書を指定します。

        戻り値
        -------
        np.ndarray
            計算されたアフィン変換行列。
        dict, optional
            変換に使用されたランダム変数の辞書 (return_random_var=Trueの場合)。
        """
        assert len(img_spacing_zyx) == 3, f"Invalid img_spacing_zyx: {img_spacing_zyx}"
        assert len(crop_center_zyx) == 3, f"Invalid center_zyx: {crop_center_zyx}"

        img_spacing_zyx = np.array(img_spacing_zyx)
        crop_center_zyx = np.array(crop_center_zyx)

        # 画像を中心に(0,0,0)に持ってくる
        shift = create_translation_matrix(-crop_center_zyx + 0.5)

        # スペーシングの正規化
        scale = create_scaling_matrix(img_spacing_zyx / self.norm_spacing_zyx)

        # ランダムにスケーリングを変える
        random_scale = random_var.get(
            "random_scale", random_scaling(self.scaling_range_zyx, is_training, rng=rng)
        )

        # ランダムに軸を反転する
        random_flip = random_var.get(
            "random_flip",
            random_flipping(self.random_flip_axis_zyx, is_training, rng=rng),
        )

        # ランダムに回転する
        random_rotate = random_var.get(
            "random_rotate", random_rotation(self.random_rot_deg_zyx, is_training, rng=rng)
        )

        # 画像中心を戻す
        reverse_shift = create_translation_matrix(self.crop_size_zyx / 2 - 0.5)

        # ランダムに画像をずらす
        random_shift = random_var.get(
            "random_shift",
            random_translation(
                self.random_shift_rate_zyx,
                self.crop_size_zyx,
                is_training,
                rng=rng,
            ),
        )

        affine_matrix = (
            shift
            @ scale
            @ random_scale
            @ random_flip
            @ random_rotate
            @ reverse_shift
            @ random_shift
        )
        if return_random_var:
            random_var_dict = {
                "random_flip": random_flip,
                "random_scale": random_scale,
                "random_rotate": random_rotate,
                "random_shift": random_shift,
            }
            return affine_matrix, random_var_dict

        return affine_matrix

    def fix_start(self, affine, shift_zyx):
        """
        アフィン変換を調整し、変換前にシフトを適用します。

        パラメータ
        ----------
        affine : np.ndarray
            アフィン変換行列。
        shift_zyx : tuple of float
            適用するシフトをz, y, xの順で指定します。

        戻り値
        -------
        np.ndarray
            シフト後のアフィン変換行列。
        """
        affine = create_translation_matrix(shift_zyx) @ affine
        return affine

    def fix_end(self, affine, shift_zyx):
        """
        アフィン変換を調整し、変換後にシフトを適用します。

        パラメータ
        ----------
        affine : np.ndarray
            アフィン変換行列。
        shift_zyx : tuple of float
            適用するシフトをz, y, xの順で指定します。

        戻り値
        -------
        np.ndarray
            シフト後のアフィン変換行列。
        """
        affine = affine @ create_translation_matrix(shift_zyx)
        return affine

    def apply(self, input_array, affine_matrix, order, cval=0, use_six_point=False):
        """
        入力データにアフィン変換を適用します。

        パラメータ
        ----------
        input_array : np.ndarrayまたはNone
            変換対象となる入力配列。
            shape=(z,y,x): 画像やマスクとして扱います。
            shape=(z,y,x,c): チャンネルでループして処理します
            shape=(3,): 点座標として扱います。
            shape=(n,3): 座標群として扱います。
            shape=(n,2,3): 矩形(bb)として扱います。
        affine_matrix : np.ndarray
            アフィン変換行列。
        order : int
            補間の順序 (デフォルトは0: 最近傍補間)。
        cval : intまたはfloat
            入力配列外の領域に使用する定数値。
        use_six_point : bool
            Trueの場合、バウンディングボックスに6点法を使用します。（球体を仮定する場合）
            そうでない回転によって矩形が大きくなってしまうので、
            何かしらの後処理で大きさを調整することをおすすめします。

        戻り値
        -------
        np.ndarrayまたはNone
            変換された配列。
        """
        if input_array is None:
            return input_array

        return self.apply_transformation(
            input_array, affine_matrix, order, cval, self.crop_size_zyx, use_six_point
        )

    def apply_transformation(
        self, input_array, affine_matrix, order, cval, output_shape, use_six_point
    ):
        shape = input_array.shape
        if len(shape) == 1 and shape[0] == 3:
            output = self.transform_coordinates(input_array, affine_matrix)
        elif len(shape) == 2 and shape[1] == 3:
            output = self.transform_coordinates(input_array, affine_matrix)
        elif len(shape) == 3 and shape[1:] == (2, 3):
            output = self.transform_bbs(input_array, affine_matrix, use_six_point)
        elif len(shape) == 3:
            output = self.transform_array(
                input_array, affine_matrix, order, cval, output_shape
            )
        elif len(shape) == 4:
            output = self.transform_array_batch(
                input_array, affine_matrix, order, cval, output_shape
            )
        else:
            raise ValueError(f"Invalid input array shape: {shape}")
        return output

    @staticmethod
    def transform_coordinates(coords, affine_matrix):
        org_shape = coords.shape
        affine_inv = np.linalg.inv(affine_matrix)
        coords = coords.reshape(-1, 3)
        coords = (affine_inv[:3, :3] @ coords.T).T + affine_inv[:3, 3]
        return coords.reshape(org_shape)

    @staticmethod
    def transform_bbs(bbs, affine_matrix, use_six_point):
        if use_six_point:
            bbs = six_point_extractor(bbs)
        else:
            bbs = eight_point_extractor(bbs)
        bbs = bbs @ np.linalg.inv(affine_matrix).T
        return two_point_extractor(bbs)

    def transform_array(self, array, affine_matrix, order, cval, output_shape):
        output_dtype = np.float32 if int(order) > 0 else array.dtype
        return affine_transform(
            array,
            affine_matrix,
            output=output_dtype,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=True,
            output_shape=output_shape,
        )

    def transform_array_batch(self, batch, affine_matrix, order, cval, output_shape):
        output_dtype = np.float32 if int(order) > 0 else batch.dtype
        transformed_batch = np.zeros(
            list(output_shape) + [batch.shape[-1]], output_dtype
        )
        for i in range(batch.shape[-1]):
            transformed_batch[..., i] = self.transform_array(
                batch[..., i], affine_matrix, order, cval, output_shape
            )
        return transformed_batch
