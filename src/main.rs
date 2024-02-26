use itertools::Itertools;
use pcp_rec_str::pcp::{Tile, PCP, PCPDir};
use pcp_rec_str::pcpseq::{ExactSequence, MidExactSequence, PCPSequence};

use core::panic;
use regex::Regex;
use std::{
    cmp::{max, min},
    collections::{BinaryHeap, HashSet, VecDeque},
    io::{BufRead, Write},
    rc::Rc,
};

fn substrings(s: &str, min_len: usize, max_len: usize) -> Vec<String> {
    let mut ret: Vec<String> = (min_len..=min(s.len(), max_len))
        .flat_map(|l: usize| -> Vec<String> {
            (0..=s.len() - l).map(|i| s[i..i + l].to_string()).collect()
        })
        .collect();
    ret.sort();
    ret.dedup();
    ret
}

#[test]
fn test_substrings() {
    let r = substrings("abcde", 2, 4);

    assert_eq!(
        r,
        vec!["ab", "abc", "abcd", "bc", "bcd", "bcde", "cd", "cde", "de"]
    );
}

fn abstract_seq(seq: &PCPSequence, min_len: usize, max_len: usize) -> Vec<PCPSequence> {
    //return vec![seq.clone()];
    match seq {
        PCPSequence::Exact(e) => {
            let ret = substrings(&e.seq, min_len, min(max_len, e.seq.len()))
                .into_iter()
                .map(|s| PCPSequence::MidExact(MidExactSequence { mid: s, dir: e.dir }))
                .collect_vec();

            // ret.append(&mut (1..e.seq.len()).flat_map(|st|
            //     (st..e.seq.len()).rev().map(|ens| PCPSequence::MidWild(MidWildSequence {
            //         front: e.seq[..st].to_string(),
            //         back: e.seq[ens..].to_string(),
            //         dir: e.dir,
            //     })).collect_vec()
            // ).collect_vec());

            ret
        }
        PCPSequence::MidWild(e) => {
            let mut ret = vec![];

            // ret.push(PCPSequence::MidExact(MidExactSequence {
            //     mid: e.front.clone(),
            //     dir: e.dir,
            // }));
            // ret.push(PCPSequence::MidExact(MidExactSequence {
            //     mid: e.back.clone(),
            //     dir: e.dir,
            // }));

            // ret.push(PCPSequence::MidWild(MidWildSequence {
            //     front: e.front[..min(e.front.len(), 5)].to_string(),
            //     back: "".to_string(),
            //     dir: e.dir,
            // }));

            // ret.push(PCPSequence::MidWild(MidWildSequence {
            //     front: e.front[..min(e.front.len(), max_len)].to_string(),
            //     back: e.back[(e.back.len() - min(e.back.len(), max_len))..].to_string(),
            //     dir: e.dir,
            // }));
            ret
            //vec![seq.clone()]
        }
        PCPSequence::MidExact(e) => {
            let ret = substrings(&e.mid, min_len, min(max_len, e.mid.len() - 1))
                .into_iter()
                .map(|s| PCPSequence::MidExact(MidExactSequence { mid: s, dir: e.dir }))
                .collect_vec();
            ret
        }
    }
}

#[derive(Debug, Clone)]
struct SuccessResult {
    max_assumption_idx: i32,
}

#[derive(Debug, Clone)]
enum TimeoutResult {
    Timeout,
    Success(SuccessResult),
    ContainsEmpty,
}

impl TimeoutResult {
    fn succeeded(&self) -> bool {
        match self {
            TimeoutResult::Success(_) => true,
            _ => false,
        }
    }
}

fn enumerate_suffices(s: &str) -> Vec<String> {
    let mut ret = vec![];
    for i in 0..s.len() {
        ret.push(s[i..].to_string());
    }
    ret
}
fn enumerate_prefices(s: &str) -> Vec<String> {
    let mut ret = vec![];
    for i in 0..s.len() {
        ret.push(s[..i].to_string());
    }
    ret
}
fn pcp_isok(pcp: &PCP) -> impl Fn(&PCPSequence) -> bool {
    let upper_suffices = pcp
        .tiles
        .iter()
        .flat_map(|t| enumerate_suffices(&t.up))
        .collect_vec();
    let upper_prefices = pcp
        .tiles
        .iter()
        .flat_map(|t| enumerate_prefices(&t.up))
        .collect_vec();
    let lower_suffices = pcp
        .tiles
        .iter()
        .flat_map(|t| enumerate_suffices(&t.dn))
        .collect_vec();
    let lower_prefices = pcp
        .tiles
        .iter()
        .flat_map(|t| enumerate_prefices(&t.dn))
        .collect_vec();
    let upper_substr_regex = Regex::new(
        format!(
            "^({})?({})*({})?$",
            upper_suffices.iter().join("|"),
            pcp.tiles.iter().map(|t| t.up.clone()).join("|"),
            upper_prefices.iter().join("|"),
        )
        .as_str(),
    )
    .unwrap();
    let lower_substr_regex = Regex::new(
        format!(
            "^({})?({})*({})?$",
            lower_suffices.iter().join("|"),
            pcp.tiles.iter().map(|t| t.dn.clone()).join("|"),
            lower_prefices.iter().join("|"),
        )
        .as_str(),
    )
    .unwrap();

    let upper_suffix_regex = Regex::new(
        format!(
            "^({})?({})*$",
            upper_suffices.iter().join("|"),
            pcp.tiles.iter().map(|t| t.up.clone()).join("|"),
        )
        .as_str(),
    )
    .unwrap();
    let lower_suffix_regex = Regex::new(
        format!(
            "^({})?({})*$",
            lower_suffices.iter().join("|"),
            pcp.tiles.iter().map(|t| t.dn.clone()).join("|"),
        )
        .as_str(),
    )
    .unwrap();

    let isok = move |s: &PCPSequence| -> bool {
        match s {
            PCPSequence::Exact(e) => {
                lower_suffix_regex.is_match(&e.seq) && upper_suffix_regex.is_match(&e.seq)
            }
            PCPSequence::MidExact(e) => {
                upper_substr_regex.is_match(&e.mid) && lower_substr_regex.is_match(&e.mid)
            }
            PCPSequence::MidWild(e) => {
                upper_substr_regex.is_match(&e.front)
                    && lower_substr_regex.is_match(&e.front)
                    && upper_suffix_regex.is_match(&e.back)
                    && lower_suffix_regex.is_match(&e.back)
            }
        }
    };
    Box::new(isok)
}

fn check_reach_empty(
    pcp: &PCP,
    s: &PCPSequence,
    assumptions: &Vec<&PCPSequence>,
    emptied: &Vec<PCPSequence>,
    max_iter: u32,
) -> bool {
    let mut q = BinaryHeap::<(i32, PCPSequence)>::new();

    q.push((
        -(match s {
            PCPSequence::Exact(e) => e.seq.len() as i32,
            PCPSequence::MidExact(e) => e.mid.len() as i32,
            PCPSequence::MidWild(e) => e.front.len() as i32 + e.back.len() as i32,
        }),
        s.clone(),
    ));

    let mut visited: HashSet<PCPSequence> = HashSet::new();
    let mut visited_exacts: HashSet<PCPSequence> = HashSet::new();

    fn is_visited(
        visited: &HashSet<PCPSequence>,
        visited_exacts: &HashSet<PCPSequence>,
        seq: &PCPSequence,
    ) -> bool {
        match seq {
            PCPSequence::Exact(_) => visited_exacts.contains(seq),
            _ => visited
                .iter()
                .chain(visited_exacts.iter())
                .any(|f| f.contains(seq)),
        }
    }

    while ((visited.len() + visited_exacts.len()) as u32) < max_iter && q.len() > 0 {
        let (_, seq) = q.pop().unwrap();
        if seq.contains_empty() {
            return true;
        }
        if is_visited(&visited, &visited_exacts, &seq) {
            continue;
        }
        if emptied.iter().any(|f| seq.contains(f)) {
            return true;
        }

        let next = seq.apply_pcp(pcp, |s| true);
        match seq {
            PCPSequence::Exact(_) => {
                visited_exacts.insert(seq);
            }
            _ => {
                visited.insert(seq);
            }
        }

        for n in next.iter() {
            if n.contains_empty() {
                return true;
            }
            if is_visited(&visited, &visited_exacts, n) {
                continue;
            }
            if emptied.iter().any(|f| n.contains(f)) {
                return true;
            }
            q.push((
                -(match n {
                    PCPSequence::Exact(e) => e.seq.len() as i32,
                    PCPSequence::MidExact(e) => e.mid.len() as i32,
                    PCPSequence::MidWild(e) => e.front.len() as i32 + e.back.len() as i32,
                }),
                n.clone(),
            ));
        }
    }

    false
}

fn enumerate_configurations(pcp: &PCP) -> Vec<ExactSequence> {
    let mut q = VecDeque::new();
    q.push_back(ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    });
    let mut visited = vec![];

    for _ in 0..10000 {
        if q.len() == 0 {
            break;
        }
        let seq = q.pop_front().unwrap();
        let next = seq.apply_pcp(pcp);
        for n in next {
            if visited.contains(&n) {
                continue;
            }
            visited.push(n.clone());
            q.push_back(n);
        }
    }

    println!("visited: {:?}", visited.len());
    visited.sort();
    visited.dedup();
    visited
}

fn enumerate_empty_configs(pcp: &PCP) -> Vec<PCPSequence> {
    let mut ret = enumerate_configurations(&pcp.swap_pcp().reverse_pcp())
        .iter()
        .map(|s| {
            PCPSequence::Exact(ExactSequence {
                seq: s.seq.chars().rev().collect::<String>(),
                dir: s.dir,
            })
        })
        .collect_vec();
    ret.sort_by_key(|s| s.num_chars());
    ret
}

struct AbstractDFSState<'a> {
    pcp: &'a PCP,
    is_ok: Rc<dyn Fn(&PCPSequence) -> bool>,
    emptied: Box<Vec<PCPSequence>>,
}
impl AbstractDFSState<'_> {
    fn new<'a>(pcp: &'a PCP) -> AbstractDFSState<'a> {
        let is_ok = pcp_isok(pcp);
        AbstractDFSState {
            pcp: pcp,
            is_ok: Rc::new(is_ok),
            emptied: Box::new(enumerate_empty_configs(pcp)),
        }
    }
}

fn check_recursive(
    cur: &PCPSequence,
    assumptions: &mut Vec<PCPSequence>,
    // conclusions[0] is theorem
    conclusions: &mut Vec<Vec<PCPSequence>>,
    depthlimit: u64,
    theorem_checking: bool,
    all_exact: bool,
    state: &mut AbstractDFSState,
    max_len: usize,
) -> TimeoutResult {
    if cur.contains_empty() {
        if all_exact {
            panic!("empty in all exact");
        }
        return TimeoutResult::ContainsEmpty;
    }
    // println!(
    //     "conclusions: {:?}",
    //     conclusions.iter().map(|s| s.len()).collect_vec()
    // );
    if state.emptied.iter().any(|f| cur.contains(f)) {
        return TimeoutResult::ContainsEmpty;
    }
    if depthlimit <= 0 {
        return TimeoutResult::Timeout;
    }

    for g in -1..assumptions.len() as i32 {
        if (g >= 0 && assumptions[g as usize].contains(cur))
            || conclusions.len() > (g + 1) as usize
                && conclusions[(g + 1) as usize]
                    .iter()
                    .any(|f| f.contains(cur))
        {
            return TimeoutResult::Success(SuccessResult {
                max_assumption_idx: g as i32,
            });
        }
    }

    println!("assumptions: {:?}", assumptions);
    println!("cur: {:?} {:?}", assumptions.len(), cur);
    let mut abstractions = abstract_seq(&cur, 1, max_len)
        .into_iter()
        .filter(|s| -> bool {
            if let Some(f) = state.emptied.iter().find(|f| s.contains(f)) {
                //println!("emptied: {:?} {:?}", s, f);
                return false;
            }
            // if !check_reach_empty(state.pcp, s, &vec![], &state.emptied, 1000) {
            //     state.emptied.push(s.clone());
            //     return false;
            // }
            true
        })
        .collect_vec();

    abstractions.sort_by_key(|s| match s {
        PCPSequence::Exact(e) => e.seq.len() as i32,
        PCPSequence::MidExact(e) => (e.mid.len() as i32) - 10000,
        PCPSequence::MidWild(e) => e.front.len() as i32 + e.back.len() as i32 - 1000,
    });

    abstractions.push((*cur).clone());
    println!("abstractions: {:?}", abstractions.len());

    let mut non_abstracted_empty = false;
    for s in abstractions.iter() {
        let is_non_abstracted = s == cur;
        let next_all_exact = all_exact
            && match s {
                PCPSequence::Exact(_) => true,
                _ => false,
            };

        assert!(s.contains(cur));
        let sofar_count = assumptions.len();

        let new_is_ok = |s: &PCPSequence| {
            (state.is_ok)(s)
            // && assumptions.iter().all(|f| !f.contains(s))
            // && conclusions.iter().flatten().all(|f| !f.contains(s))
        };

        // if check_reach_empty(state.pcp, s, &mut vec![], &state.emptied, 100) {
        //     state.emptied.push((*s).clone());
        //     if is_non_abstracted {
        //         non_abstracted_empty = true;
        //     }
        //     //println!("skipped {:?}", s);
        //     continue;
        // }

        let mut nexts: Vec<PCPSequence> = s.apply_pcp_avoid_midwild(&state.pcp, new_is_ok);
        println!("nexts: {:?}", nexts.len());

        //refine_recursive_seqs(&mut nexts);

        nexts.sort_by_key(|s| (s.num_chars() as i32));
        //nexts.shuffle(&mut rand::thread_rng());
        assumptions.push(s.clone());
        println!("refined nexts: {:?}", nexts.len());

        let mut nexts_all_ok: TimeoutResult = TimeoutResult::Success(SuccessResult {
            max_assumption_idx: -1,
        });
        for nex in nexts.iter() {
            let nex_res = check_recursive(
                nex,
                assumptions,
                conclusions,
                depthlimit - 1,
                theorem_checking,
                next_all_exact,
                state,
                max_len,
            );

            match nex_res {
                TimeoutResult::Timeout => {
                    nexts_all_ok = TimeoutResult::Timeout;
                    break;
                }
                TimeoutResult::ContainsEmpty => {
                    state.emptied.push(s.clone());
                    if is_non_abstracted {
                        non_abstracted_empty = true;
                    }
                    nexts_all_ok = TimeoutResult::ContainsEmpty;
                    break;
                }
                TimeoutResult::Success(r) => {
                    nexts_all_ok = TimeoutResult::Success(SuccessResult {
                        max_assumption_idx: if let TimeoutResult::Success(cr) = nexts_all_ok {
                            max(r.max_assumption_idx as i32, cr.max_assumption_idx as i32)
                        } else {
                            panic!("not success")
                        },
                    });
                }
            }
        }

        while conclusions.len() <= sofar_count + 1 {
            conclusions.push(vec![]);
        }
        let l = conclusions[sofar_count + 1].clone();
        let lMidExact: Vec<&PCPSequence> = l
            .iter()
            .filter(|s| match s {
                PCPSequence::MidExact(_) => true,
                _ => false,
            })
            .collect_vec();
        match nexts_all_ok {
            TimeoutResult::Timeout => {
                if lMidExact.len() > 0 {
                    println!("lost {:?}", lMidExact);
                    // for s in lMidExact.iter() {
                    //     // if check_reach_empty(state.pcp, s, &vec![], &state.emptied, 10000) {
                    //     //     println!("lost reachable {:?}", s);
                    //     //     state.emptied.push((*s).clone());
                    //     // } else {
                    //     //     println!("lost unreachable {:?}", s);
                    //     //     // let mut newAss = assumptions.iter().take(sofar_count).map(|s| s.clone()).collect_vec();
                    //     //     // let mut newConc = conclusions.iter().take(sofar_count + 1).map(|s| s.clone()).collect_vec();
                    //     //     // let rr = check_recursive(
                    //     //     //     s,
                    //     //     //     &mut newAss,
                    //     //     //     &mut newConc,
                    //     //     //     depthlimit * 2,
                    //     //     //     false,
                    //     //     //     false,
                    //     //     //     state,
                    //     //     // );
                    //     //     // println!("lost unreachable repo {:?} {:?}", s, rr);
                    //     //     // if rr.succeeded() {
                    //     //     //     state.emptied.push((*s).clone());
                    //     //     // }
                    //     // }
                    // }
                }
            }
            TimeoutResult::ContainsEmpty => {
                if lMidExact.len() > 0 {
                    println!("empty lost {:?}", lMidExact);
                }
            }
            TimeoutResult::Success(r) => {
                let addedd_idx: usize = min(r.max_assumption_idx + 1, sofar_count as i32) as usize;
                println!("added_idx: {:?} vs {:?}", sofar_count, addedd_idx);
                conclusions[addedd_idx].extend(vec![s.clone()]);
                conclusions[addedd_idx].extend(l.clone());
                if lMidExact.len() > 0 {
                    println!(
                        "conclusionSizes: {:?}",
                        conclusions
                            .iter()
                            .take(20)
                            .map(|c| c
                                .iter()
                                .filter(|s| if let PCPSequence::MidExact(_) = s {
                                    true
                                } else {
                                    false
                                })
                                .collect_vec())
                            .collect_vec()
                    );
                    println!("level: {:?} brought: {:?}", addedd_idx, lMidExact);
                }
                conclusions.truncate(sofar_count + 1);
                assumptions.truncate(sofar_count);
                return TimeoutResult::Success(SuccessResult {
                    max_assumption_idx: (addedd_idx as i32) - 1,
                });
            }
        }
        assumptions.truncate(sofar_count);
        conclusions.truncate(sofar_count + 1);
    }

    if non_abstracted_empty {
        state.emptied.push((*cur).clone());
        TimeoutResult::ContainsEmpty
    } else {
        TimeoutResult::Timeout
    }
}

#[derive(Debug, Clone)]
struct Result {
    assumptions: Vec<PCPSequence>,
    result: bool,
    start: PCPSequence,
}

fn pdr_like(pcp: &PCP, max_len: usize) -> (bool, Vec<Result>) {
    log::debug!("pcp: {:?}", pcp);
    let mut firsts = ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    }
    .apply_pcp(&pcp);
    log::debug!("firsts: {:?}", firsts);

    let mut state = AbstractDFSState::new(pcp);

    let mut results = vec![];
    let mut theorems = vec![
        // PCPSequence::MidExact(MidExactSequence { mid: "011111110".to_string(), dir: PCPDir::DN }),
        // PCPSequence::MidExact(MidExactSequence { mid: "0111111111110".to_string(), dir: PCPDir::DN }),
        // PCPSequence::MidExact(MidExactSequence { mid: "0111111110".to_string(), dir: PCPDir::UP }),
        // PCPSequence::MidExact(MidExactSequence { mid: "010".to_string(), dir: PCPDir::DN }),
        // PCPSequence::MidExact(MidExactSequence { mid: "0110".to_string(), dir: PCPDir::DN }),
        // PCPSequence::MidExact(MidExactSequence { mid: "011111101111110".to_string(), dir: PCPDir::DN }),
        // PCPSequence::MidExact(MidExactSequence { mid: "01110".to_string(), dir: PCPDir::UP }),
    ];
    for s in firsts {
        let mut has_ok = false;
        for max_depth_log in 20..30 {
            let max_depth = 1 << max_depth_log;
            println!("max_depth: {:?}", max_depth);
            let mut assumptions = vec![];
            let mut conclusions = vec![theorems.clone()];
            let r = check_recursive(
                &PCPSequence::Exact(s.clone()),
                &mut assumptions,
                &mut conclusions,
                max_depth,
                false,
                true,
                &mut state,
                max_len,
            );
            println!("start: {:?} result: {:?}", s.seq, r.succeeded());
            println!("assumptions: {:?}", assumptions);
            println!("conclusions: {:?}", conclusions);
            theorems.extend(conclusions.iter().flatten().cloned());
            if r.succeeded() {
                results.push(Result {
                    assumptions: conclusions.into_iter().flatten().collect_vec(),
                    result: r.succeeded(),
                    start: PCPSequence::Exact(s.clone()),
                });
                has_ok = true;
                break;
            }
            match r {
                TimeoutResult::Timeout => {
                    println!("timeout");
                }
                TimeoutResult::ContainsEmpty => {
                    panic!("failed");
                }
                _ => {}
            }
        }
        if !has_ok {
            return (false, results);
        }
    }
    (true, results)
}

fn enumerate_substrings_from_pcp(pcp: &PCP, max_len: usize) -> (Vec<String>, Vec<String>) {
    let mut q = VecDeque::new();
    q.push_back(ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    });
    let mut visited = vec![];

    for _ in 0..20000 {
        let seq = q.pop_front().unwrap();
        let next = seq.apply_pcp(pcp);
        for n in next {
            if visited.contains(&n) {
                continue;
            }
            visited.push(n.clone());
            q.push_back(n);
        }
    }

    visited.sort();
    visited.dedup();

    let mut upper_substrs = vec![];
    let mut lower_substrs = vec![];

    for s in visited {
        if s.dir == PCPDir::UP {
            upper_substrs.append(&mut substrings(&s.seq, 0, max_len));
        } else {
            lower_substrs.append(&mut substrings(&s.seq, 0, max_len));
        }
    }

    upper_substrs.sort();
    upper_substrs.dedup();
    lower_substrs.sort();
    lower_substrs.dedup();

    (upper_substrs, lower_substrs)
}

fn parse_file(f: &str) -> Vec<(String, PCP)> {
    let mut ret = vec![];
    let f = std::fs::File::open(f).unwrap();
    let lines = std::io::BufReader::new(f).lines().collect_vec();
    for result in lines.iter() {
        let record = result.as_ref().unwrap();
        let pcp = PCP::parse_pcp_string(&record.as_str());
        ret.push((record.to_string(), pcp));
    }
    ret
}

fn find_nonempty_substrs(pcp: &PCP) -> (Vec<String>, Vec<String>) {
    let substr_len = 20;
    let (upper_substrs, lower_substrs) = enumerate_substrings_from_pcp(&pcp, substr_len);
    let mut upper_trues: Vec<String> = vec![];

    let emptied = enumerate_empty_configs(pcp);

    for s in upper_substrs
        .into_iter()
        .filter(|s| -> bool { s.len() < substr_len })
        .sorted_by_key(|s| s.len())
    {
        if upper_trues.iter().any(|t| s.contains(t)) {
            continue;
        }
        let result = check_reach_empty(
            &pcp,
            &&PCPSequence::MidExact(MidExactSequence {
                mid: s.clone(),
                dir: PCPDir::UP,
            }),
            &mut vec![],
            &emptied,
            100000,
        );
        if !result {
            println!("upper true: {:?}", s);
            upper_trues.push(s);
        }
    }

    let mut lower_trues: Vec<String> = vec![];
    for s in lower_substrs
        .into_iter()
        .filter(|s| -> bool { s.len() < substr_len })
        .sorted_by_key(|s| s.len())
    {
        if lower_trues.iter().any(|t| s.contains(t)) {
            continue;
        }
        let result = check_reach_empty(
            &pcp,
            &PCPSequence::MidExact(MidExactSequence {
                mid: s.clone(),
                dir: PCPDir::DN,
            }),
            &mut vec![],
            &emptied,
            100000,
        );
        if !result {
            println!("lower true: {:?}", s);
            lower_trues.push(s);
        }
    }

    (upper_trues, lower_trues)
}

fn get_nonempty_for_pcps() {
    for (idx, (raw, pcp)) in parse_file("a.csv").iter().enumerate() {
        println!("pcp: {:?} {:?}", idx, pcp);
        let (upper_trues, lower_trues) = find_nonempty_substrs(&pcp);
        std::fs::write(
            "results/".to_string() + &idx.to_string() + ".txt",
            format!("{:?}\n{:?}\n{:?}\n", raw, upper_trues, lower_trues),
        )
        .unwrap();

        let (upper_trues, lower_trues) = find_nonempty_substrs(&pcp.reverse_pcp().swap_pcp());
        std::fs::write(
            "results/".to_string() + &idx.to_string() + "-rev.txt",
            format!("{:?}\n{:?}\n{:?}\n", raw, upper_trues, lower_trues),
        )
        .unwrap();
    }
}

fn from_input() {
    let mut input_str = String::new();
    std::io::stdin().read_line(&mut input_str).unwrap();
    let pcp = PCP::parse_pcp_string(&input_str);
    println!("pcp: {:?}", pcp);

    let (upper_trues, lower_trues) = find_nonempty_substrs(&pcp);
    println!("upper recursive strings: {:?}", upper_trues);
    println!("lower recursive strings: {:?}", lower_trues);
}

fn find_answer(pcp: &PCP) -> bool {
    let mut q: VecDeque<ExactSequence> = VecDeque::new();
    let s = ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    };
    q.push_back(s);
    let mut visited: HashSet<ExactSequence> = HashSet::new();

    for _ in 0..10000 {
        if q.len() == 0 {
            break;
        }
        let seq = &q.pop_front().unwrap();
        let next = seq.apply_pcp(pcp);
        for n in next {
            if visited.contains(&n) {
                continue;
            }
            if n.seq.len() == 0 {
                return true;
            }

            visited.insert(n.clone());
            q.push_back(n);
        }
    }
    false
}
/*
Format of an instance:
   1st line: index
   2nd line: size, width, optimal length, number of optimal solutions
   3rd & 4th lines: pairs
   5th line: a line break

Instance 1:
3 3 75 2
110   1     0
1     0     110
*/
fn parse_instance_list(fname: &str) -> Vec<PCP> {
    let mut ret = vec![];
    let f = std::fs::File::open(fname).unwrap();
    let mut content = std::io::BufReader::new(f)
        .lines()
        .collect_vec()
        .iter()
        .map(|s| s.as_ref().unwrap().to_string())
        .collect_vec();
    let seperator = Regex::new(r"\s+").expect("Invalid regex");

    while content.is_empty() == false {
        let startline = content
            .iter()
            .enumerate()
            .find(|(_, s)| s.starts_with("Instance"))
            .unwrap()
            .0;
        let ups = seperator.split(&content[startline + 2]).collect_vec();
        let dns = seperator.split(&content[startline + 3]).collect_vec();

        ret.push(PCP {
            tiles: ups
                .iter()
                .zip(dns.iter())
                .map(|(u, d)| Tile {
                    up: u.to_string(),
                    dn: d.to_string(),
                })
                .filter(|t| -> bool { t.up.len() > 0 && t.dn.len() > 0 })
                .collect_vec(),
        });
        content = content[startline + 5..].to_vec();
    }

    ret
}

fn apply_pdr_single(idx: usize, pcp: &PCP, raw: &str, rev: bool, max_len: usize) {
    let mut pcp = pcp;
    let reversed = pcp.reverse_pcp().swap_pcp();
    if rev {
        pcp = &reversed;
    }

    println!("{idx} reversed:{rev} pcp: {:?}", raw);
    let (res, results) = pdr_like(&pcp, max_len);
    println!("result: {:?}", res);

    let mut file = std::fs::File::create(
        (if res {
            "proofs/".to_string()
        } else {
            "inv/".to_string()
        }) + &idx.to_string()
            + "-"
            + &rev.to_string()
            + ".txt",
    )
    .unwrap();
    file.write(format!("pcp: {:?}\n", raw).as_bytes()).unwrap();
    file.write(format!("pcp: {:?}\n", pcp).as_bytes()).unwrap();

    file.write(format!("reversed: {:?}\n", rev).as_bytes())
        .unwrap();
    file.write(format!("result: {:?}\n", res).as_bytes())
        .unwrap();

    if res == false {
        for r in results {
            file.write(format!("{:?}\n", r).as_bytes()).unwrap();
        }
    } else {
        let results = results
            .into_iter()
            .map(|r| refine_proof(&pcp, r))
            .collect_vec();
        for r in results {
            file.write(format!("{:?}\n", r).as_bytes()).unwrap();
        }
    }
}

fn apply_pdr() {
    let instances = parse_file("a.csv");

    let args: Vec<String> = std::env::args().collect();

    let instance_idx = args[1].parse::<usize>().unwrap();
    let rev = args[2].parse::<bool>().unwrap();

    let max_len = args[3].parse::<usize>().unwrap();

    let (raw, pcp) = &instances[instance_idx];

    let mut pcp = &PCP::parse_pcp_string("Tile(110,1)Tile(1,01)Tile(0,110)");
    let raw = "Tile(110,1)Tile(1,01)Tile(0,110)";
    println!("pcp: {:?}", pcp);

    apply_pdr_single(instance_idx, &pcp, &raw, rev, max_len);
}

fn check_valid_proof(pcp: &PCP, result: &Result) -> bool {
    if result.assumptions.iter().any(|p| p.contains_empty()) {
        return false;
    }
    if result
        .assumptions
        .iter()
        .all(|p| !p.contains(&result.start))
    {
        return false;
    }
    let is_ok = pcp_isok(pcp);
    for p in result.assumptions.iter() {
        let nexts = p.apply_pcp_avoid_midwild(pcp, Box::new(&is_ok));

        for n in nexts {
            if result
                .assumptions
                .iter()
                .any(|p: &PCPSequence| p.contains(&n))
            {
                continue;
            }
            return false;
        }
    }
    true
}

fn refine_recursive_seqs(seqs: &mut Vec<PCPSequence>) {
    for i in (0..seqs.len()).rev() {
        if seqs
            .iter()
            .enumerate()
            .any(|(j, s)| i != j && s.contains(&seqs[i]))
        {
            seqs.remove(i);
        }
    }
}

fn refine_proof(pcp: &PCP, result: Result) -> Result {
    let mut ret: Vec<PCPSequence> = result.assumptions.clone();

    if !check_valid_proof(pcp, &result) {
        println!(
            "contains_empty: {:?}",
            result.assumptions.iter().any(|s| s.contains_empty())
        );
        panic!("invalid proof");
    }
    println!("valid");

    for (idx, a) in result.assumptions.iter().enumerate().rev() {
        ret.remove(idx);
        if !check_valid_proof(
            pcp,
            &Result {
                assumptions: ret.clone(),
                ..result.clone()
            },
        ) {
            ret.insert(idx, a.clone());
        }
    }
    Result {
        assumptions: ret.clone(),
        ..result
    }
}

fn hard_instances_check() {
    let pcps = parse_instance_list("200hard.txt");
    for pcp in pcps.iter().skip(9) {
        println!("pcp: {:?}", pcp);
        let (res, results) = pdr_like(pcp, 0);
        println!("result: {:?}", res);
        if res {
            let refined = refine_proof(&pcp, results[0].clone());
            println!("refined: {:?}", refined);
            println!(
                "{:?} -> {:?}",
                results[0].assumptions.len(),
                refined.assumptions.len()
            );
            panic!("fail");
        }
    }
}

fn check_sanity() {
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(100,1), Tile(0,100), Tile(1,00)))");
    println!("pcp: {:?}", pcp);
    let (res, results) = pdr_like(&pcp, 20);
    println!("result: {:?}", res);
    if res {
        let refined = refine_proof(&pcp, results[0].clone());
        println!("refined: {:?}", refined);
        println!(
            "{:?} -> {:?}",
            results[0].assumptions.len(),
            refined.assumptions.len()
        );
        panic!("fail");
    }
}

fn find_recursive_strings(pcp: &PCP) -> Vec<PCPSequence> {
    let (us, ds) = find_nonempty_substrs(&pcp);

    let mut ss = us
        .iter()
        .map(|s| {
            PCPSequence::MidExact(MidExactSequence {
                mid: s.clone(),
                dir: PCPDir::UP,
            })
        })
        .chain(ds.iter().map(|s: &String| {
            PCPSequence::MidExact(MidExactSequence {
                mid: s.clone(),
                dir: PCPDir::DN,
            })
        }))
        .collect_vec();

    let mut ret_theorems: Vec<PCPSequence> = vec![];
    loop {
        let mut allok = true;
        ret_theorems = vec![];
        for (idx, s) in ss.iter().enumerate() {
            let mut others = vec![];
            for (i, o) in ss.iter().enumerate() {
                if i != idx {
                    others.push(o.clone());
                }
            }
            let mut state = AbstractDFSState::new(&pcp);
            let mut conclusions = vec![others.clone()];
            println!("{:?} vs {:?}", s, others);
            for max_depth_log in 0..10 {
                let max_depth = 1 << max_depth_log;
                let res = check_recursive(
                    &s,
                    &mut vec![],
                    &mut conclusions,
                    max_depth,
                    false,
                    false,
                    &mut state,
                    300,
                );
                println!("{:?}", res);
                match res {
                    TimeoutResult::Timeout => {
                        println!("timeout");
                        if max_depth_log == 6 {
                            allok = false;
                            break;
                        }
                        continue;
                    }
                    TimeoutResult::ContainsEmpty => {
                        println!("contains empty");
                        allok = false;
                        break;
                    }
                    TimeoutResult::Success(_) => {}
                }
            }
            if !allok {
                println!("failed: {:?}", s);
                ss.remove(idx);
                break;
            }
            ret_theorems.extend(conclusions.iter().flatten().cloned());
        }
        if allok {
            println!("recursive strings: {:?}", ss);
            println!("all ok");
            return ret_theorems;
        }
    }
}
fn find_recursive_strings_for_pcp() {
    let instances = parse_file("a.csv");
    let input = std::env::args().collect_vec();
    let idx = input[1].parse::<usize>().unwrap();
    let rev = input[2].parse::<bool>().unwrap();
    let mut pcp = &instances[idx].1;

    let rev_pcp = pcp.reverse_pcp().swap_pcp();
    let raw = &instances[idx].0;

    if rev {
        pcp = &rev_pcp;
    }

    println!("pcp: {:?} idx: {idx} rev: {rev}", pcp);
    let res = find_recursive_strings(&pcp);

    let mut file = std::fs::File::create(
        "results/".to_string() + &idx.to_string() + "-" + &rev.to_string() + ".txt",
    )
    .unwrap();
    file.write(format!("pcp: {:?}\n", raw).as_bytes()).unwrap();
    file.write(format!("pcp: {:?}\n", pcp).as_bytes()).unwrap();
    file.write(format!("reversed: {:?}\n", rev).as_bytes())
        .unwrap();
    file.write(format!("result: {:?}\n", res).as_bytes())
        .unwrap();
}

fn reduce_checking(pcp: &PCP, iter: usize) -> bool {
    let mut reduced_aut = pcp.to_automaton();
    for i in 0..iter {
        let n_reduced_aut = reduced_aut.construct_reduced_automaton();
        assert!(n_reduced_aut
            .upper
            .get_input_nfa()
            .is_equal(&n_reduced_aut.lower.get_input_nfa()));
        if n_reduced_aut
            .upper
            .get_input_nfa()
            .is_equal(&reduced_aut.upper.get_output_nfa())
            && n_reduced_aut
                .upper
                .get_output_nfa()
                .is_equal(&reduced_aut.upper.get_input_nfa())
        {
            reduced_aut = n_reduced_aut;

            break;
        }

        assert!(reduced_aut
            .upper
            .get_output_nfa()
            .includes(&n_reduced_aut.upper.get_input_nfa()));
        assert!(reduced_aut
            .lower
            .get_output_nfa()
            .includes(&n_reduced_aut.lower.get_input_nfa()));
        println!(
            "{:?}",
            n_reduced_aut
                .upper
                .get_input_nfa()
                .is_equal(&reduced_aut.upper.get_output_nfa())
        );
        println!(
            "{:?}",
            n_reduced_aut
                .upper
                .get_output_nfa()
                .is_equal(&reduced_aut.upper.get_input_nfa())
        );
        println!(
            "{:?}",
            n_reduced_aut
                .lower
                .get_input_nfa()
                .is_equal(&reduced_aut.lower.get_output_nfa())
        );
        println!(
            "{:?}",
            n_reduced_aut
                .lower
                .get_output_nfa()
                .is_equal(&reduced_aut.lower.get_input_nfa())
        );

        reduced_aut = n_reduced_aut;
        assert!(reduced_aut.upper.get_input_nfa().accept(&vec![]));
        assert!(reduced_aut.lower.get_input_nfa().accept(&vec![]));
        let upper_size = reduced_aut
            .upper
            .transition
            .values()
            .flatten()
            .collect_vec()
            .len();
        let lower_size = reduced_aut
            .lower
            .transition
            .values()
            .flatten()
            .collect_vec()
            .len();
        println!(
            "iter: {:?} upper: {:?} lower: {:?}",
            i, upper_size, lower_size
        );
    }

    reduced_aut.upper.show_dot("reduced/upper");
    reduced_aut.lower.show_dot("reduced/lower");

    let upper_size = reduced_aut
        .upper
        .transition
        .values()
        .flatten()
        .collect_vec()
        .len();
    let lower_size = reduced_aut
        .lower
        .transition
        .values()
        .flatten()
        .collect_vec()
        .len();
    return upper_size == 0 || lower_size == 0;
}

#[test]
fn test_reduced_checking() {
    assert!(reduce_checking(&PCP::parse_pcp_string("Tile(100,1)"), 10));
    assert!(!reduce_checking(&PCP::parse_pcp_string("Tile(1,1)"), 10));
    assert!(!reduce_checking(
        &PCP::parse_pcp_string("Tile(1,1)Tile(0,1)"),
        4
    ));
    assert!(!reduce_checking(
        &PCP::parse_pcp_string("Tile(01,1)Tile(1,10)"),
        4
    ));
    assert!(reduce_checking(
        &PCP::parse_pcp_string("Tile(100,1)Tile(10,1)"),
        10
    ));
    assert!(reduce_checking(
        &PCP::parse_pcp_string("Tile(11011,10110)Tile(011,1)"),
        4
    ));
    assert!(!reduce_checking(
        &PCP::parse_pcp_string("Tile(11011,10110)Tile(011,1)Tile(1,1101)"),
        4
    ));
}
fn sanity_check_reduce_pcp_aut() {
    let pcps = parse_file("a.csv").into_iter().map(|s| s.1).collect_vec();
    //let pcps = parse_instance_list("200hard.txt");
    let input = std::env::args().collect_vec();
    let idx = input[1].parse::<usize>().unwrap();
    let rev = input[2].parse::<bool>().unwrap();

    let pcp = &pcps[idx];
    let revpcp = pcp.reverse_pcp();
    let pcp = if rev { &revpcp } else { pcp };

    println!("pcp: {:?} rev:{:?}", pcp, rev);
    let res: bool = reduce_checking(pcp, 10);
    println!("result: {:?}", res);
}

fn main() {
    //hard_instances_check();
    //sanity_check_reduce_pcp_aut();
    //println!("Hello, world!");
    //find_recursive_strings_for_pcp()

    //check_sanity();
    //let pcps = parse_file("a.csv");
    // let r = check_recursive(&pcps[0].1.reverse_pcp().swap_pcp(),
    //     &PCPSequence::MidExact(MidExactSequence{
    //         mid: "1111111111".to_string(),
    //         dir: PCPDir::UP,
    //     }), &mut HashMap::new(),
    // &mut vec![
    //     PCPSequence::MidExact(MidExactSequence{
    //         mid: "01111110".to_string(),
    //         dir: PCPDir::UP,
    //     }),
    //     PCPSequence::MidExact(MidExactSequence{
    //         mid: "0110".to_string(),
    //         dir: PCPDir::UP,
    //     })
    // ],
    //&mut Vec::new(), 0, true);
    //println!("result: {:?}", r.succeeded());
    //get_nonempty_for_pcps();
    //hard_instances_check();
    // let lorents = parse_pcp_string("PCP(Vector(Tile(10,0), Tile(0,001), Tile(001,1)))");
    // apply_pdr_single(123, &lorents, "PCP(Vector(Tile(10, 0), Tile(0, 001), Tile(001, 1)))", false);
    //apply_pdr_single(21, &pcps[21].1, &pcps[21].0, false);

    apply_pdr();

    // apply_pdr(false);
    // for _ in 0..1000 {
    //     let pcp = gen_random_pcp(3, 4);
    //     println!("{:?}", pcp);

    //     if find_answer(&pcp) {
    //         println!("found!");

    //         let res = pdr_like(pcp);
    //         if res {
    //             panic!("fail");
    //         }
    //     }
    // }
    //get_recs_for_pcps();
    //from_input();
    //explor_main(true);

    //pdr_like(parse_pcp_string("PCP(Vector(Tile(1111,110), Tile(1011,1), Tile(1,1111)))"));
    // let pcp = parse_pcp_string("PCP(Vector(Tile(1111,110), Tile(1110,1), Tile(1,1111)))");
    // pdr_like(swap_pcp(&reverse_pcp(&pcp)));

    // let pcp = parse_pcp_string("PCP(Vector(Tile(1110,01), Tile(1101,11), Tile(1,1011)))");
    // let result = check_recursive(&pcp, "111011110", PCPDir::UP);
    // println!("result: {:?}", result);
}
