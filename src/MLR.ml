
module A = BatArray
module L = BatList
module S = BatString

open Printf

(* load all lines from given CSV file
   [skip_header] allows to ignore the first line (CSV header), if necessary.
   The first column is supposed to contain the target variable.
   All other columns are supposed to contain the parameters whose weights
   in the MLR model we want to find. *)
let load_csv_file skip_header sep csv_fn =
  let all_lines = match Utls.lines_of_file csv_fn with
    | [] -> failwith ("MLR.load_csv_file: no lines in " ^ csv_fn)
    | x :: xs -> if skip_header then xs else x :: xs in
  let fst_line = L.hd all_lines in
  let dimx = (S.count_char fst_line sep) + 1 in
  let dimy = L.length all_lines in
  let res = A.make_matrix dimx dimy 0.0 in
  L.iteri (fun y line ->
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

type std_param = { mean: float;
                   sd: float }

type std_params = std_param array

type model_param = { mean: float;
                     sd: float;
                     w: float }

type model = model_param array

(* compute the standardization parameters *)
let standardization_params arr =
  let dimx = A.length arr in
  let res = A.make dimx { mean = 0.0; sd = 0.0 } in
  (* the first column (arr.(0)) is the target value;
     it must not be normalized *)
  for x = 1 to dimx - 1 do
    let col = arr.(x) in
    let mean = A.favg col in
    let sd = Utls.stddev_a col in
    res.(x) <- { mean; sd }
  done;
  res

(* train model
   !!! the features in [arr] must be already normalized !!! *)
let train_model _arr =
  failwith "not implemented yet"

let apply_model _model _arr =
  failwith "not implemented yet"
