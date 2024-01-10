// use itertools::Itertools;

// use std::collections::HashSet;
// use rand::Rng;
// type State = u64;

// fn random_state() -> State {
//     rand::thread_rng().gen()
// }

// 4
// #[derive(Debug, Clone)]
// pub struct Transition {
//     from: State,
//     to: State,
//     in_sym: Option<char>,
//     out_sym: Option<char>,
// }
// pub struct Transducer {
//     pub transitions: Vec<Transition>,
//     pub initial_state: State,
//     pub final_states: HashSet<State>,
// }

// impl Transducer {
//     fn union(&self, rhs: &Transducer) -> Transducer {
//         let initial_state = random_state();
//         let additional_transitions = vec![
//             Transition {
//                 from: initial_state,
//                 to: self.initial_state,
//                 in_sym: None,
//                 out_sym: None,
//             },
//             Transition {
//                 from: initial_state.clone(),
//                 to: rhs.initial_state.clone(),
//                 in_sym: None,
//                 out_sym: None,
//             },
//         ];
//         let mut result = Transducer {
//             transitions: [self.transitions, rhs.transitions, additional_transitions].concat(),
//             initial_state,
//             final_states: self.final_states.union(&rhs.final_states).cloned().collect(),
//         };
//         result
//     }

//     fn concat(&self, rhs: &Transducer) -> Transducer {
//         let mut result = Transducer {
//             transitions: [self.transitions.clone(), rhs.transitions.clone()].concat(),
//             initial_state: self.initial_state.clone(),
//             final_states: rhs.final_states.clone(),
//         };
//         for state in self.final_states.iter() {
//             result.transitions.push(Transition {
//                 from: state.clone(),
//                 to: rhs.initial_state.clone(),
//                 in_sym: None,
//                 out_sym: None,
//             });
//         }
//         result
//     }

//     fn intersects(&self, rhs: &Transducer) -> Transducer {
//         let mut result = Transducer {
//             transitions: Vec::new(),
//             initial_state: self.initial_state ^ rhs.initial_state,
//             final_states: self.final_states,
//         };

        
//     }
// }