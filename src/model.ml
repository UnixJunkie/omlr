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
  (* TODO train on train, test on test *)
  (* TODO compute R2 *)
  (* TODO gnuplot *)
  (* TODO implement --NxCV *)
  let input_fn = CLI.get_string ["-i"] args in
  CLI.finalize (); (* ------------------------------------------------------ *)
  let train_lines, _test_lines =
    let all_lines = MLR.read_csv_file ~randomize ~skip_header input_fn in
    Cpm.Utls.train_test_split !train_portion all_lines in
  let train_data = MLR.matrix_of_csv_lines ',' train_lines in
  let std_params = MLR.standardization_params train_data in
  MLR.standardize std_params train_data;
  let coeffs = MLR.train_model ~debug train_data in
  let coeffs_str = Utls.string_of_floats_array coeffs in
  Log.info "model coeffs: %s" coeffs_str;
  ()

let () = main ()
