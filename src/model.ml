(* Copyright (C) 2020, Francois Berenger

   Yamanishi laboratory,
   Department of Bioscience and Bioinformatics,
   Faculty of Computer Science and Systems Engineering,
   Kyushu Institute of Technology,
   680-4 Kawazu, Iizuka, Fukuoka, 820-8502, Japan.

   Train/test a MLR regressor *)

open Printf

module A = Array
module CLI = Minicli.CLI
module L = BatList
module Log = Dolog.Log
module MLR = Omlr.MLR
module Utls = Omlr.Utls

let main () =
  Log.(set_log_level DEBUG);
  Log.color_on ();
  Log.info "start";
  let argc, args = CLI.init () in
  let show_help = CLI.get_set_bool ["-h";"--help"] args in
  if argc = 1 || show_help then
    begin
      eprintf "usage:\n\
               %s\n  \
               [-i <input.csv>]: input CSV file\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [-s|--save <filename>]: save model to file\n  \
               [-l|--load <filename>]: restore model from file\n  \
               [-o <filename>]: predictions output file\n  \
               [--no-plot]: don't call gnuplot\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this message\n"
        Sys.argv.(0);
      exit 1
    end;
  (* defaults options ------------------------------------------------------ *)
  let randomize = true in
  let skip_header = true in
  let train_portion = ref 0.8 in
  let debug = false in
  let sep = ',' in
  (* TODO gnuplot *)
  (* TODO implement --NxCV *)
  (* TODO implement -l *)
  (* TODO implement -s *)
  let input_fn = CLI.get_string ["-i"] args in
  CLI.finalize (); (* ------------------------------------------------------ *)
  let train_lines, test_lines =
    let all_lines = MLR.read_csv_file ~randomize ~skip_header input_fn in
    Cpm.Utls.train_test_split !train_portion all_lines in
  let nb_train, nb_test = L.(length train_lines, length test_lines) in
  Log.info "train: %d test: %d total: %d"
    nb_train nb_test (nb_train + nb_test);
  (* train *)
  let train_data = MLR.matrix_of_csv_lines ~sep train_lines in
  let model = MLR.train_model ~debug train_data in
  (* FBR: log the model in some way; even if not saved to file *)
  (* test *)
  let actual, test_data =
    let a = MLR.matrix_of_csv_lines ~sep test_lines in
    (* first col. is target value *)
    (a.(0), Utls.transpose_matrix a) in
  let predicted = A.make nb_test 0.0 in
  assert(A.length test_data = nb_test); (* FBR: CHECK *)
  for i = 0 to nb_test - 1 do
    predicted.(i) <- MLR.predict_one model test_data.(i)
  done;
  let r2 = Cpm.RegrStats.r2 (A.to_list actual) (A.to_list predicted) in
  Log.info "R2: %.3f" r2

let () = main ()
