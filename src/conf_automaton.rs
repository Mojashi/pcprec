use std::borrow::Borrow;

use itertools::Itertools;

use crate::{
    automaton::{self, Transducer, Transition},
    pcp::PCP,
    pcpseq::PCPDir,
};

pub struct PCPConf {
    dir: PCPDir,
    conf: automaton::Transducer<char, char>,
}

pub struct PCPAutomaton {
    // these domains are the same
    upper: automaton::Transducer<char, char>,
    lower: automaton::Transducer<char, char>,
}

#[test]
fn test_reduced_aut() {
    let pcp = PCP::parse_pcp_string("Tile(100,1), Tile(0,100), Tile(1,00)");
    let ans = "0200211".chars().collect::<Vec<char>>();
    let co_ans = "1001100100100".chars().collect::<Vec<char>>();

    let mut reduced_aut = pcp.to_automaton();

    reduced_aut.upper.show_dot("reduced/upper");
    reduced_aut.lower.show_dot("reduced/lower");

    for i in 0..4 {
        reduced_aut = reduced_aut.construct_reduced_automaton();

        println!("reduced upper: {}", reduced_aut.upper.states.len());
        println!("reduced lower: {}", reduced_aut.lower.states.len());
        reduced_aut
            .upper
            .show_dot(("reduced/upper".to_string() + i.to_string().as_str()).borrow());
        reduced_aut
            .lower
            .show_dot(("reduced/lower".to_string() + i.to_string().as_str()).borrow());
        if i % 2 == 0 {
            assert!(reduced_aut.upper.get_input_nfa().accept(co_ans.borrow()));
            assert!(reduced_aut.lower.get_input_nfa().accept(co_ans.borrow()));
        } else {
            assert!(reduced_aut.upper.get_input_nfa().accept(ans.borrow()));
            assert!(reduced_aut.lower.get_input_nfa().accept(ans.borrow()));
        }
        assert!(reduced_aut.upper.get_input_nfa().accept(&vec![]));
        assert!(reduced_aut.lower.get_input_nfa().accept(&vec![]));
    }
}

#[test]
fn test_reduced_aut2() {
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,1), Tile(1,1110), Tile(0,111)))");
    //let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(110,1), Tile(1,0), Tile(0,100)))");
    let mut reduced_aut = pcp.to_automaton();
    for i in 0..6 {
        reduced_aut = reduced_aut.construct_reduced_automaton();
        assert!(reduced_aut.upper.get_input_nfa().accept(&vec![]));
        assert!(reduced_aut.lower.get_input_nfa().accept(&vec![]));
        
        println!("reduced upper: {}", reduced_aut.upper.transition.values().flatten().collect_vec().len());
        println!("reduced lower: {}", reduced_aut.lower.transition.values().flatten().collect_vec().len());
    }
    reduced_aut.upper.show_dot("reduced/upper");
    reduced_aut.lower.show_dot("reduced/lower");
    println!("ans: {:?}", reduced_aut.upper.get_input_nfa().get_element());
    println!("ans: {:?}", reduced_aut.lower.get_input_nfa().get_element());
}

impl PCPAutomaton {
    pub fn construct_reduced_automaton(&self) -> PCPAutomaton {
        let upper_inverse = self.upper.inverse();
        let lower_inverse = self.lower.inverse();

        let upper_inverse_input = upper_inverse.get_input_nfa();
        let lower_inverse_input = lower_inverse.get_input_nfa();

        let upper_inverse_reduced = upper_inverse
            .intersection_input(&lower_inverse_input)
            .remove_none_none_transitions()
            .reduce_size_unreachable();
        let lower_inverse_reduced = lower_inverse
            .intersection_input(&upper_inverse_input)
            .remove_none_none_transitions()
            .reduce_size_unreachable();

        return PCPAutomaton {
            upper: upper_inverse_reduced,
            lower: lower_inverse_reduced,
        };
    }
}

impl PCP {
    pub fn to_automaton(&self) -> PCPAutomaton {
        let upper = self.to_automaton_upper();
        let lower = self.swap_pcp().to_automaton_upper();
        return PCPAutomaton { upper, lower };
    }

    fn to_automaton_upper(&self) -> automaton::Transducer<char, char> {
        let mut aut = Transducer::new();

        let start = aut.start.clone();
        aut.accept.insert(start.clone());
        for (idx, tile) in self.tiles.iter().enumerate() {
            assert!(idx.to_string().len() == 1);
            let idx_char = idx.to_string().chars().next().unwrap();

            let mut last_state = start.clone();
            for (i, t) in tile.up.chars().enumerate() {
                let next_state = if i < tile.up.len() - 1 {
                    automaton::new_state()
                } else {
                    start.clone()
                };
                let label = if i == 0 {
                    (Some(idx_char), Some(t))
                } else {
                    (None, Some(t))
                };
                aut.add_transition(Transition {
                    from: last_state.clone(),
                    to: next_state.clone(),
                    label: label,
                });
                last_state = next_state;
            }
        }
        aut
    }
}
