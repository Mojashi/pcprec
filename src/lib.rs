pub mod automaton;
pub mod conf_automaton;
pub mod pcp;
pub mod pcpseq;
pub mod union_pdr;
pub mod union_pdr2;
pub mod suffix_tree;

pub mod union_find;

use wasm_bindgen::prelude::*;


#[wasm_bindgen]
pub fn wasm_test() -> String {
    "Hello from Rust!".to_string()
}