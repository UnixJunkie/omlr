
module A = BatArray
module Arr = Owl.Arr
module L = BatList
module Linalg = Owl.Linalg
module Mat = Owl.Mat
module S = BatString

open Printf

(* load all lines from given CSV file
   [skip_header] allows to ignore the first line (CSV header), if necessary *)
let load_csv_file skip_header sep csv_fn =
  let all_lines = match Utls.lines_of_file csv_fn with
    | [] -> failwith ("MLR.load_csv_file: no lines in " ^ csv_fn)
    | x :: xs -> if skip_header then xs else x :: xs in
  let fst_line = L.hd all_lines in
  let dimx = (S.count_char fst_line sep) + 1 in
  let dimy = L.length all_lines in
  let res = A.make_matrix dimx dimy 0.0 in
  L.iteri (fun (y: int) (line: string) ->
      let tokens = L.map float_of_string (S.split_on_char sep line) in
      let n = L.length tokens in
      (if n <> dimx then
         failwith
           (sprintf "MLR.load_csv_file: file %s line %d has %d features \
                     instead of %d" csv_fn y n dimx)
      );
      L.iteri (fun x feat ->
          res.(x).(y) <- feat
        ) tokens
    ) all_lines;
  res

type norm_param = { mean: float;
                    std: float }

(* compute the normalization parameters,
 * apply them on the array and return them with the modified array *)
let normalize_features arr =
  let dimx = A.length arr in
  let dimy = A.length arr.(0) in
  (* the first column is the target value; it must not be normalized *)
  let norm_params = A.make (dimx - 1) { mean = 0.0; std = 0.0 } in
  let to_normalize = A.copy arr in
  for x = 1 to dimx - 1 do
    let column = A.init dimy (fun y -> arr.(x).(y)) in
    let mean = Owl.Stats.mean column in
    let std = Owl.Stats.std ~mean column in
    norm_params.(x - 1) <- { mean; std};
    for y = 0 to dimy - 1 do
      let z = to_normalize.(x).(y) in
      to_normalize.(x).(y) <- (z -. mean) /. std
    done
  done;
  (norm_params, to_normalize)

(* compute model using gradient descent
   !!! the features in [arr] must be already normalized !!! *)
let train_model_gd arr =
  let dimx = A.length arr in
  let data = Mat.of_arrays arr in
  (* all lines, only first column (target value) *)
  let y = Mat.get_slice [[]; [0]] data in
  (* all lines, all but first column *)
  let x = Mat.get_slice [[]; L.range 1 `To (dimx - 1)] data in
  Owl.Regression.D.ols ~i:true y x

(* compute model analytically (may fail on some problem instances)
   !!! the features in [arr] must be already normalized !!! *)
let train_model_ana arr =
  let dimx = A.length arr in
  let dimy = A.length arr.(0) in
  let data = Mat.of_arrays arr in
  (* all lines, only first column (target value) *)
  let y = Mat.get_slice [[]; [0]] data in
  (* all lines, all but first column *)
  let x = Mat.get_slice [[]; L.range 1 `To (dimx - 1)] data in
  let z =
    let o = Arr.ones [|dimy; 1|] in
    Arr.concatenate ~axis:1 [|o; x|] in
  let zT = Mat.transpose z in
  let zTz_inv =
    let zTz = Mat.dot zT z in
    Linalg.D.inv zTz in
  Mat.(dot (dot zTz_inv zT) y)

let apply_model _norm_params _model _arr =
  failwith "not implemented yet"
