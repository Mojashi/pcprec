use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet, VecDeque},
    fmt::format,
    hash::Hash,
    str::Chars,
    vec,
};

use itertools::Itertools;

#[derive(Debug)]
enum AppRegex {
    Star(Box<AppRegex>),
    Or(Box<AppRegex>, Box<AppRegex>),
    Concat(Box<AppRegex>, Box<AppRegex>),
    Ch(char),
    Eps,
}

#[test]
fn test_parse() {
    let regex = AppRegex::parse("a(bddd|cf)*d");
    println!("{:?}", regex);
}

impl AppRegex {
    pub fn parse(s: &str) -> AppRegex {
        /*
        R -> R | R
        R -> RR
        R -> R*
        R -> (R)
        R -> a
        R -> ε
        */

        fn parse_inner(iter: &mut Chars, until: Option<char>) -> AppRegex {
            let mut stack: Vec<AppRegex> = vec![];
            stack.push(AppRegex::Eps);

            while let Some(c) = iter.next() {
                if until == Some(c) {
                    return stack
                        .into_iter()
                        .reduce(|a, b| AppRegex::Concat(Box::new(a), Box::new(b)))
                        .unwrap();
                }
                match c {
                    '*' => {
                        let last = stack.pop().unwrap();
                        stack.push(AppRegex::Star(Box::new(last)));
                    }
                    '(' => {
                        let s = parse_inner(iter, Some(')'));
                        stack.push(s);
                    }
                    ')' => {
                        panic!("Unmatched )");
                    }
                    '|' => {
                        let last = stack.pop().unwrap();
                        let s = parse_inner(iter, until);
                        stack.push(AppRegex::Or(Box::new(last), Box::new(s)));
                    }
                    _ => {
                        let last = stack.pop().unwrap();
                        stack.push(AppRegex::Concat(Box::new(last), Box::new(AppRegex::Ch(c))));
                    }
                }
            }
            return stack
                .into_iter()
                .reduce(|a, b| AppRegex::Concat(Box::new(a), Box::new(b)))
                .unwrap();
        }
        parse_inner(&mut s.chars(), None)
    }
}

type State = String;
fn product_state(a: &State, b: &State) -> State {
    format!("({},{})", a, b)
}

#[derive(Debug, Clone)]
pub struct Transition<A>
where
    A: Eq + std::hash::Hash + Clone,
{
    pub from: State,
    pub to: State,
    pub label: A,
}

fn extract_states<A>(transitions: &Vec<Transition<A>>) -> HashSet<State>
where
    A: Eq + std::hash::Hash + Clone,
{
    let mut states = HashSet::new();
    for transition in transitions {
        states.insert(transition.from.clone());
        states.insert(transition.to.clone());
    }
    states
}

static mut STATE_COUNTER: u64 = 0;
pub fn new_state() -> State {
    unsafe {
        STATE_COUNTER += 1;
        format!("q{}", STATE_COUNTER)
    }
}
fn group_transitions_by_from<A>(
    transitions: Vec<Transition<A>>,
) -> HashMap<State, Vec<Transition<A>>>
where
    A: Eq + std::hash::Hash + Clone,
{
    transitions.into_iter().into_group_map_by(|t| t.from.clone())
}

trait HasEps {
    fn eps() -> Self;
}

#[test]
fn test_input_nfa() {
    let mut t = Transducer::new();
    t.add_transition(Transition {
        from: "q1".to_string(),
        to: "q1".to_string(),
        label: (Some('1'), Some('0')),
    });

    let nfa = t.get_input_nfa();
    nfa.show_dot("test_input_nfa");
}

#[derive(Debug)]
pub struct BaseAutomaton<A>
where
    A: Eq + std::hash::Hash + Clone + HasEps + std::fmt::Debug,
{
    pub transition: HashMap<State, Vec<Transition<A>>>,
    pub accept: HashSet<State>,
    pub start: State,
    pub states: HashSet<State>,
}

fn escape_state_for_dot(s: &State) -> String {
    format!("\"{}\"", s)
}

#[test]
fn test_reachable() {
    let mut nfa = NFA::new();
    nfa.add_transition(Transition {
        from: nfa.start.clone(),
        to: "a2".to_string(),
        label: Some('a'),
    });
    nfa.add_transition(Transition {
        from: "a2".to_string(),
        to: "a3".to_string(),
        label: Some('b'),
    });
    nfa.add_transition(Transition {
        from: "a3".to_string(),
        to: nfa.start.clone(),
        label: Some('c'),
    });
    nfa.add_transition(Transition {
        from: nfa.start.clone(),
        to: nfa.start.clone(),
        label: Some('d'),
    });
    nfa.accept = vec![nfa.start.clone()].into_iter().collect();
    assert!(nfa.reachable_states() == vec![nfa.start.clone(), "a2".to_string(), "a3".to_string()].into_iter().collect());

    let rev = nfa.reversed();
    let rev_reachable = rev.reachable_states();
    println!("{:?}", rev.transition);

    rev.show_dot("test_reachable2");
    nfa.show_dot("test_reachable");
    println!("{:?}", rev_reachable);
    assert!(
        vec![nfa.start.clone(), "a2".to_string(), "a3".to_string()].into_iter()
        .all(|s| rev_reachable.contains(&s)));
}

impl<A> BaseAutomaton<A>
where
    A: Eq + std::hash::Hash + Clone + HasEps + std::fmt::Debug,
{
    pub fn new() -> BaseAutomaton<A> {
        let start = new_state();
        BaseAutomaton {
            transition: HashMap::new(),
            accept: HashSet::new(),
            start: start.clone(),
            states: vec![start.clone()].into_iter().collect(),
        }
    }

    pub fn init(
        transition: HashMap<State, Vec<Transition<A>>>,
        accept: HashSet<State>,
        start: State,
    ) -> BaseAutomaton<A> {
        let mut states = extract_states(
            &transition
                .clone()
                .into_iter()
                .flat_map(|(_, transitions)| transitions.into_iter())
                .collect(),
        );
        states.insert(start.clone());
        let accept = accept
            .clone()
            .into_iter()
            .filter(|s| states.contains(s))
            .collect();
        BaseAutomaton {
            transition,
            accept,
            start,
            states,
        }
    }

    pub fn add_transition(&mut self, transition: Transition<A>) {
        self.states.insert(transition.from.clone());
        self.states.insert(transition.to.clone());
        self.transition
            .entry(transition.from.clone())
            .or_insert(vec![])
            .push(transition);
    }

    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph {\n");
        for (from, transitions) in &self.transition {
            for transition in transitions {
                let label = format!("{:?}", transition.label);
                dot.push_str(&format!(
                    "{} -> {} [label=\"{}\"]\n",
                    escape_state_for_dot(from),
                    escape_state_for_dot(&transition.to),
                    label
                ));
            }
        }
        for state in &self.accept {
            dot.push_str(&format!(
                "{} [shape=doublecircle]\n",
                escape_state_for_dot(state)
            ));
        }

        let super_start = new_state();
        dot.push_str(&format!("{} [shape=point]\n", super_start));
        dot.push_str(&format!(
            "{} -> {}\n",
            super_start,
            escape_state_for_dot(&self.start)
        ));

        dot.push_str("}\n");
        dot
    }
    pub fn show_dot(&self, base_name: &str) {
        let dot = self.to_dot();
        let dot_name = format!("{}.dot", base_name);
        let path = std::path::Path::new(&dot_name);
        std::fs::write(path, dot).unwrap();
        let output = std::process::Command::new("dot")
            .arg("-Tpng")
            .arg(dot_name)
            .arg("-o")
            .arg(format!("{}.png", base_name))
            .output()
            .unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }
    pub fn union(&self, other: &BaseAutomaton<A>) -> BaseAutomaton<A> {
        let other = other.rename_states();
        let mut new_transitions: Vec<Transition<A>> = self
            .transition
            .iter()
            .chain(other.transition.iter())
            .flat_map(|(_, transitions)| transitions.clone().into_iter())
            .collect_vec();
        let new_start = new_state();
        new_transitions.push(Transition {
            from: new_start.clone(),
            to: self.start.clone(),
            label: A::eps(),
        });
        new_transitions.push(Transition {
            from: new_start.clone(),
            to: other.start.clone(),
            label: A::eps(),
        });

        BaseAutomaton::init(
            group_transitions_by_from(new_transitions),
            self.accept
                .clone()
                .into_iter()
                .chain(other.accept.clone().into_iter())
                .collect::<HashSet<_>>(),
            new_start,
        )
    }

    pub fn rename_states(&self) -> BaseAutomaton<A> {
        let mut new_states = HashMap::new();
        for state in self.states.iter() {
            new_states.insert(state, new_state());
        }
        let mut new_transitions: HashMap<State, Vec<Transition<A>>> = self
            .transition
            .iter()
            .map(|(from, transitions)| {
                (
                    new_states.get(&from).unwrap().clone(),
                    transitions
                        .iter()
                        .map(|t| Transition {
                            from: new_states.get(&t.from).unwrap().clone(),
                            to: new_states.get(&t.to).unwrap().clone(),
                            label: t.label.clone(),
                        })
                        .collect(),
                )
            })
            .collect::<HashMap<_, _>>();

        BaseAutomaton {
            transition: new_transitions,
            accept: self
                .accept
                .iter()
                .map(|a| new_states.get(a).unwrap().clone())
                .collect(),
            start: new_states.get(&self.start).unwrap().clone(),
            states: self
                .states
                .iter()
                .map(|s| new_states.get(s).unwrap().clone())
                .collect(),
        }
    }

    pub fn product_by<B, C>(
        &self,
        other: &BaseAutomaton<B>,
        map_pair: impl Fn(&A, &B) -> Option<C>,
    ) -> BaseAutomaton<C>
    where
        B: Eq + std::hash::Hash + Clone + std::fmt::Debug + HasEps,
        C: Eq + std::hash::Hash + Clone + std::fmt::Debug + HasEps,
    {
        let mut product_transitions: Vec<Transition<C>> = vec![];

        let mut queue: VecDeque<(State, State)> = VecDeque::new();
        let mut inserted: HashSet<(State, State)> = HashSet::new();
        inserted.insert((self.start.clone(), other.start.clone()));
        queue.push_back((self.start.clone(), other.start.clone()));

        while !queue.is_empty() {
            let (cur_a, cur_b) = queue.pop_front().unwrap();

            let emp_transitions_a: Vec<Transition<A>> = vec![Transition {
                from: cur_a.clone(),
                to: cur_a.clone(),
                label: A::eps(),
            }];
            let emp_transitions_b: Vec<Transition<B>> = vec![Transition {
                from: cur_b.clone(),
                to: cur_b.clone(),
                label: B::eps(),
            }];
            let a_emp = vec![];
            let b_emp = vec![];

            let transitions_a = self
                .transition
                .get(&cur_a)
                .unwrap_or(&a_emp)
                .into_iter()
                .chain(emp_transitions_a.iter());
            let transitions_b = other
                .transition
                .get(&cur_b)
                .unwrap_or(&b_emp)
                .into_iter()
                .chain(emp_transitions_b.iter());

            transitions_a
                .cartesian_product(transitions_b)
                .filter_map(|(a, b)| map_pair(&a.label, &b.label).map(|label| (a, b, label)))
                .for_each(|(a, b, new_label)| {
                    let new_state = product_state(&a.to, &b.to);
                    product_transitions.push(Transition {
                        from: product_state(&cur_a, &cur_b),
                        to: new_state.clone(),
                        label: new_label,
                    });
                    if !inserted.contains(&(a.to.clone(), b.to.clone())) {
                        inserted.insert((a.to.clone(), b.to.clone()));
                        queue.push_back((a.to.clone(), b.to.clone()));
                    }
                });
        }

        product_transitions.retain(|t| t.label != C::eps() || t.from != t.to);

        BaseAutomaton::init(
            group_transitions_by_from(product_transitions),
            self.accept
                .iter()
                .cartesian_product(other.accept.iter())
                .map(|(a, b)| product_state(a, b))
                .collect(),
            product_state(&self.start, &other.start),
        )
    }

    pub fn reversed(&self) -> BaseAutomaton<A> {
        let mut new_transitions: Vec<Transition<A>> = self
            .transition
            .iter()
            .flat_map(|(_, transitions)| transitions.clone().into_iter())
            .map(|t| Transition {
                to: t.from.clone(),
                from: t.to.clone(),
                label: t.label.clone(),
            })
            .collect();

        let new_start = new_state();
        for state in self.accept.iter() {
            new_transitions.push(Transition {
                from: new_start.clone(),
                to: state.clone(),
                label: A::eps(),
            });
        }

        BaseAutomaton::init(
            group_transitions_by_from(new_transitions),
            vec![self.start.clone()].into_iter().collect(),
            new_start,
        )
    }

    pub fn reachable_states(&self) -> HashSet<State> {
        let mut reachable: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<State> = VecDeque::new();
        queue.push_back(self.start.clone());
        reachable.insert(self.start.clone());
        while let Some(cur) = queue.pop_front() {
            if let Some(transitions) = self.transition.get(&cur) {
                for transition in transitions {
                    if !reachable.contains(&transition.to) {
                        reachable.insert(transition.to.clone());
                        queue.push_back(transition.to.clone());
                    }
                }
            }
        }
        reachable
    }

    pub fn reduce_size_unreachable(&self) -> BaseAutomaton<A> {
        let forward_reachable: HashSet<State> = self.reachable_states();
        let backward_reachable: HashSet<State> = self.reversed().reachable_states();
        let reachable: HashSet<State> = forward_reachable
            .intersection(&backward_reachable)
            .cloned()
            .collect();
        let new_transitions = self
            .transition
            .iter()
            .flat_map(|(_, transitions)| transitions.clone().into_iter())
            .filter(|t| reachable.contains(&t.from) && reachable.contains(&t.to))
            .collect();
        BaseAutomaton::init(
            group_transitions_by_from(new_transitions),
            self.accept.clone(),
            self.start.clone(),
        )
    }

    // A -(none)-> B -(none)-> C => A -(none)-> C
    pub fn remove_none_none_transitions(&self) -> BaseAutomaton<A> {
        let target_to_map: HashMap<State, Vec<&Transition<A>>> = self
            .transition
            .values()
            .flatten()
            .into_group_map_by(|t| t.to.clone());

        let mut removed_something: bool = false;
        let mut new_transitions: Vec<Transition<A>> =
            self.transition.values().flatten().cloned().collect();
        for state in self.states.iter() {
            if *state == self.start || self.accept.contains(state) {
                continue;
            }
            if let Some(out_transitions) = self.transition.get(state) {
                if let Some(in_transitions) = target_to_map.get(state) {
                    if out_transitions.iter().all(|t| t.label == A::eps()) {
                        if in_transitions.iter().all(|t| t.label == A::eps()) {
                            removed_something = true;
                            new_transitions.retain(|t| t.to != *state && t.from != *state);
                            new_transitions.extend(
                                in_transitions
                                    .iter()
                                    .cartesian_product(out_transitions.iter())
                                    .map(|(i, o)| Transition {
                                        from: i.from.clone(),
                                        to: o.to.clone(),
                                        label: A::eps(),
                                    }),
                            );
                        }
                    }
                }
            }
        }
        let ret = BaseAutomaton::init(
            group_transitions_by_from(new_transitions),
            self.accept.clone(),
            self.start.clone(),
        );
        if removed_something {
            ret.remove_none_none_transitions()
        } else {
            ret
        }
    }
}

impl<A, B> HasEps for (Option<A>, Option<B>)
where
    A: Eq + std::hash::Hash + Clone,
    B: Eq + std::hash::Hash + Clone,
{
    fn eps() -> (Option<A>, Option<B>) {
        (None, None)
    }
}

pub type Transducer<A, B> = BaseAutomaton<(Option<A>, Option<B>)>;

impl<A, B> Transducer<A, B>
where
    A: Eq + std::hash::Hash + Clone + std::fmt::Debug,
    B: Eq + std::hash::Hash + Clone + std::fmt::Debug,
{
    pub fn inverse(&self) -> Transducer<B, A> {
        let new_transitions: Vec<Transition<(Option<B>, Option<A>)>> = self
            .transition
            .iter()
            .flat_map(|(_, transitions)| {
                transitions
                    .iter()
                    .map(|t| Transition {
                        to: t.to.clone(),
                        from: t.from.clone(),
                        label: (t.label.1.clone(), t.label.0.clone()),
                    })
                    .collect::<Vec<_>>()
            })
            .collect_vec();
        Transducer::init(
            group_transitions_by_from(new_transitions),
            self.accept.clone(),
            self.start.clone(),
        )
    }
    pub fn get_input_nfa(&self) -> NFA<A> {
        let transitions = self.transition
            .iter()
            .flat_map(|(_, transitions)| transitions.clone().into_iter())
            .map(|t| Transition {
                from: t.from.clone(),
                to: t.to.clone(),
                label: t.label.0,
            });
        NFA::init (
            group_transitions_by_from(transitions.collect()),
            self.accept.clone(),
            self.start.clone(),
        )
    }

    pub fn intersection_input(&self, nfa: &NFA<A>) -> Transducer<A, B> {
        self.product_by::<Option<A>, (Option<A>, Option<B>)>(nfa, |a, b| {
            if a.0 == *b {
                Some(a.clone())
            } else {
                None
            }
        })
    }

    pub fn compose<C: Eq + std::hash::Hash + Clone + std::fmt::Debug + HasEps>(
        &self,
        other: &Transducer<B, C>,
    ) -> Transducer<A, C> {
        todo!()
    }
}

impl<A> HasEps for Option<A>
where
    A: Eq + std::hash::Hash + Clone,
{
    fn eps() -> Option<A> {
        None
    }
}

pub type NFA<A> = BaseAutomaton<Option<A>>;

#[test]
fn test_from_regex() {
    let regex = AppRegex::parse("a(b|c)*d");
    println!("{:?}", regex);
    let nfa = NFA::from_regex(&regex);
    nfa.show_dot("test_from_regex");
    println!("{:?}", nfa);
    assert!(nfa.accept(&vec!['a', 'd']));
    assert!(nfa.accept(&vec!['a', 'b', 'd']));
    assert!(nfa.accept(&vec!['a', 'c', 'd']));
    assert!(nfa.accept(&vec!['a', 'b', 'b', 'd']));
    assert!(nfa.accept(&vec!['a', 'c', 'c', 'd']));
    assert!(nfa.accept(&vec!['a', 'b', 'b', 'b', 'd']));
    assert!(nfa.accept(&vec!['a', 'c', 'c', 'c', 'd']));
    assert!(nfa.accept(&vec!['a', 'b', 'c', 'd']));
    assert!(nfa.accept(&vec!['a', 'c', 'b', 'd']));
    assert!(nfa.accept(&vec!['a', 'b', 'b', 'c', 'd']));
    assert!(nfa.accept(&vec!['a', 'c', 'c', 'b', 'd']));
    assert!(!nfa.accept(&vec!['a', 'b', 'b', 'd', 'b', 'd']));
}

impl NFA<char> {
    pub fn from_regex(regex: &AppRegex) -> NFA<char> {
        match regex {
            AppRegex::Star(regex) => {
                let mut nfa: NFA<char> = NFA::from_regex(regex);
                nfa.accept.clone().iter().for_each(|state| {
                    nfa.add_transition(Transition {
                        from: state.clone(),
                        to: nfa.start.clone(),
                        label: None,
                    });
                });
                nfa.add_transition(Transition {
                    from: nfa.start.clone(),
                    to: nfa.accept.iter().next().unwrap().clone(),
                    label: None,
                });
                nfa
            }
            AppRegex::Or(left, right) => {
                let left_nfa = NFA::from_regex(left);
                let right_nfa = NFA::from_regex(right);
                left_nfa.union(&right_nfa)
            }
            AppRegex::Concat(left, right) => {
                let mut left_nfa = NFA::from_regex(left);
                let right_nfa = NFA::from_regex(right);
                left_nfa.transition.extend(right_nfa.transition);

                left_nfa.accept.clone().iter().for_each(|state| {
                    left_nfa.add_transition(Transition {
                        from: state.clone(),
                        to: right_nfa.start.clone(),
                        label: None,
                    });
                });

                left_nfa.accept = right_nfa.accept;
                left_nfa
            }
            AppRegex::Ch(c) => {
                let mut nfa = NFA::new();
                nfa.start = new_state();
                nfa.accept = vec![new_state()].into_iter().collect();
                nfa.add_transition(Transition {
                    from: nfa.start.clone(),
                    to: nfa.accept.iter().next().unwrap().clone(),
                    label: Some(c.clone()),
                });
                nfa
            }
            AppRegex::Eps => {
                let start = new_state();
                NFA::init(
                    HashMap::new(),
                    vec![start.clone()].into_iter().collect(),
                    start,
                )
            }
        }
    }
}

impl<A> NFA<A>
where
    A: Eq + std::hash::Hash + Clone + std::fmt::Debug,
{
    fn accept_from_state<'a>(
        &'a self,
        s: Vec<&A>,
        state: &'a State,
        visited: &mut HashSet<(usize, &'a State)>,
    ) -> bool {
        if visited.contains(&(s.len(), state)) {
            return false;
        }
        if s.len() == 0 && self.accept.contains(state) {
            return true;
        }
        visited.insert((s.len(), state));

        if let Some(transition) = self.transition.get(state) {
            for transition in transition {
                if let Some(label) = &transition.label {
                    if s.len() > 0 && s[0] == label {
                        if self.accept_from_state(s[1..].to_vec(), &transition.to, visited) {
                            return true;
                        }
                    }
                } else {
                    if self.accept_from_state(s.clone(), &transition.to, visited) {
                        return true;
                    }
                }
            }
        }
        false
    }
    pub fn accept(&self, s: &Vec<A>) -> bool {
        let mut visited: HashSet<(usize, &State)> = HashSet::new();
        self.accept_from_state(s.iter().collect(), &self.start, &mut visited)
    }

    pub fn intersection(&self, other: &NFA<A>) -> NFA<A> {
        self.product_by(other, |a, b| if a == b { Some(a.clone()) } else { None })
    }

    pub fn get_element(&self) -> Option<Vec<A>> {
        let mut visited: HashSet<&State> = HashSet::new();

        fn dfs<'a, A>(
            nfa: &'a NFA<A>,
            cur: &'a State,
            visited: &mut HashSet<&'a State>,
        ) -> Option<Vec<A>>
        where
            A: Eq + std::hash::Hash + Clone + std::fmt::Debug,
        {
            if visited.contains(cur) {
                return None;
            }
            visited.insert(cur);
            if nfa.accept.contains(cur) {
                return Some(vec![]);
            }
            if let Some(transition) = nfa.transition.get(cur) {
                for transition in transition {
                    if let Some(s) = dfs(nfa, &transition.to, visited) {
                        if let Some(label) = &transition.label {
                            return Some(
                                vec![label.clone()]
                                    .into_iter()
                                    .chain(s.into_iter())
                                    .collect(),
                            );
                        }
                        return Some(s);
                    }
                }
            }
            None
        }

        dfs(self, &self.start, &mut visited)
    }

    pub fn is_empty(&self) -> bool {
        self.get_element().is_none()
    }

    // pub fn reduce_size_bisimulation(&self) -> NFA<A> {
    //     let mut partition: Vec<HashSet<State>> = vec![self.accept.clone(), self.states.clone()];

    // }
}
