mod pcp;
mod pcpseq;

use itertools::Itertools;
use pcp::{parse_pcp_string, Tile, PCP};
use pcpseq::{MidExactSequence, PCPSequence};
use rand::seq::SliceRandom;
use regex::Regex;

use std::{
    cmp::min,
    collections::{BinaryHeap, HashSet, VecDeque},
    io::{BufRead, Write},
};

use crate::pcpseq::{ExactSequence, MidWildSequence, PCPDir};

fn substrings(s: &str, max_len: usize) -> Vec<String> {
    let mut ret: Vec<String> = (0..min(s.len(), max_len + 1))
        .flat_map(|l| -> Vec<String> {
            (0..s.len() - l).map(|i| s[i..i + l].to_string()).collect()
        })
        .collect();
    ret.sort();
    ret.dedup();
    ret
}

fn abstract_seq(seq: &PCPSequence, max_len: usize) -> Vec<PCPSequence> {
    match seq {
        PCPSequence::Exact(e) => {
            let mut ret = substrings(&e.seq, max_len)
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

            ret.push(seq.clone());
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

            ret.push(PCPSequence::MidWild(MidWildSequence {
                front: e.front[..min(e.front.len(), 5)].to_string(),
                back: "".to_string(),
                dir: e.dir,
            }));

            // ret.push(PCPSequence::MidWild(MidWildSequence {
            //     front: e.front[..min(e.front.len(), max_len)].to_string(),
            //     back: e.back[(e.back.len() - min(e.back.len(), max_len))..].to_string(),
            //     dir: e.dir,
            // }));
            ret.push(seq.clone());
            ret
            //vec![seq.clone()]
        }
        PCPSequence::MidExact(e) => {
            let mut ret = substrings(&e.mid, max_len)
                .into_iter()
                .map(|s| PCPSequence::MidExact(MidExactSequence { mid: s, dir: e.dir }))
                .collect_vec();
            ret.push(seq.clone());
            ret
        }
    }
}

#[derive(Debug, Clone)]
enum TimeoutResult {
    Timeout,
    Result(bool),
}

impl TimeoutResult {
    fn succeeded(&self) -> bool {
        match self {
            TimeoutResult::Timeout => false,
            TimeoutResult::Result(r) => *r,
        }
    }
}

fn check_reach_empty(
    pcp: &PCP,
    s: &PCPSequence,
    assumptions: &mut Vec<PCPSequence>,
    emptied: &Vec<PCPSequence>,
    max_iter: u32,
) -> bool {
    let mut q = BinaryHeap::<(i32, PCPSequence)>::new();

    q.push((-(s.num_chars() as i32), s.clone()));

    let mut visited: HashSet<PCPSequence> = HashSet::new();
    visited.extend(assumptions.iter().cloned());

    while (visited.len() as u32) < max_iter && q.len() > 0 {
        let (_, seq) = q.pop().unwrap();
        if seq.contains_empty() {
            return true;
        }
        if visited.contains(&seq) {
            continue;
        }
        if emptied.iter().any(|f| seq.contains(f)) {
            return true;
        }
        visited.insert(seq.clone());

        let next = seq.apply_pcp(pcp);

        for n in next.iter() {
            if n.contains_empty() {
                return true;
            }
            if visited.contains(&n) {
                continue;
            }
            if emptied.iter().any(|f| n.contains(f)) {
                return true;
            }
            q.push((-(n.num_chars() as i32), n.clone()));
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
    let ret = enumerate_configurations(&pcp.swap_pcp().reverse_pcp())
        .iter()
        .map(|s| {
            PCPSequence::Exact(ExactSequence {
                seq: s.seq.chars().rev().collect::<String>(),
                dir: s.dir,
            })
        })
        .collect_vec();
    ret
}

struct AbstractDFSState<'a> {
    pcp: &'a PCP,
    bad_theorems: Box<HashSet<PCPSequence>>,
    theorems: Box<Vec<PCPSequence>>,
    emptied: Box<Vec<PCPSequence>>,
    failed_theorems: Box<Vec<PCPSequence>>,
}
impl AbstractDFSState<'_> {
    fn new<'a>(pcp: &'a PCP) -> AbstractDFSState<'a> {
        AbstractDFSState {
            pcp: pcp,
            bad_theorems: Box::new(HashSet::new()),
            theorems: Box::new(vec![]),
            emptied: Box::new(enumerate_empty_configs(pcp)),
            failed_theorems: Box::new(vec![]),
        }
    }
}

fn check_recursive(
    cur: &PCPSequence,
    assumptions: &mut Vec<PCPSequence>,
    depthlimit: u64,
    theorem_checking: bool,
    all_exact: bool,
    state: &mut AbstractDFSState,
) -> TimeoutResult {
    if cur.contains_empty() {
        if all_exact {
            panic!("empty in all exact");
        }
        return TimeoutResult::Result(false);
    }
    // if all_exact {
    //     println!("all exact {:?}", depthlimit);
    // }
    if depthlimit <= 0 {
        //println!("timeout {:?} {:?}", cur, theorem_checking);
        return TimeoutResult::Timeout;
    }
    if state.emptied.iter().any(|f| cur.contains(f)) {
        //println!("emptied! {:?}", emptied.iter().find(|f| cur.contains(f)));
        return TimeoutResult::Result(false);
    }
    //println!("cur: {:?}", cur, );
    //println!("{:?}", assumptions.len());

    if assumptions
        .iter()
        .chain(state.theorems.iter())
        .any(|s| -> bool { s.contains(cur) })
    {
        //println!("assumption hit!");
        return TimeoutResult::Result(true);
    }

    let abstractions = abstract_seq(&cur, 10)
        .into_iter()
        .filter(|s| -> bool {
            !s.contains_empty()
                && state.emptied.iter().all(|f| !s.contains(f))
                && assumptions
                    .iter()
                    .chain(state.theorems.iter())
                    .all(|f| !f.contains(s))
        })
        .collect_vec();

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

        let mut nexts: Vec<PCPSequence> = s.apply_pcp(&state.pcp);
        refine_recursive_seqs(&mut nexts);
        nexts.shuffle(&mut rand::thread_rng());
        assumptions.push(s.clone());
        log::debug!("nexts: {:?}", nexts);
        let ress = nexts
            .iter()
            .map(|nex| {
                check_recursive(
                    nex,
                    assumptions,
                    depthlimit - 1,
                    theorem_checking,
                    next_all_exact,
                    state,
                )
            })
            .collect_vec();

        let has_empty = ress.iter().any(|r| match r {
            TimeoutResult::Timeout => false,
            TimeoutResult::Result(r) => *r == false,
        });

        if has_empty {
            state.emptied.append(
                &mut s
                    .sample()
                    .iter()
                    .take(10)
                    .map(|a| {
                        PCPSequence::Exact(ExactSequence {
                            seq: a.clone(),
                            dir: s.dir(),
                        })
                    })
                    .collect_vec(),
            );
            if is_non_abstracted {
                non_abstracted_empty = true;
            }
        }

        let res = ress.iter().all(|r| r.succeeded());
        if !res && !has_empty && depthlimit > 10 {
            // timeout
            match s {
                PCPSequence::Exact(_) => {}
                PCPSequence::MidExact(e) => {
                    //println!("promising {:?}", s);
                    if !theorem_checking && !state.bad_theorems.contains(s) && e.mid.len() <= 15 {
                        let mut tmp_assumptions: Vec<PCPSequence> = vec![];
                        let mut new_state = AbstractDFSState {
                            pcp: state.pcp,
                            theorems: state.theorems.clone(),
                            bad_theorems: state.bad_theorems.clone(),
                            emptied: state.emptied.clone(),
                            failed_theorems: state.failed_theorems.clone(),
                        };
                        let is_theorem = check_recursive(
                            s,
                            &mut tmp_assumptions,
                            40,
                            true,
                            false,
                            &mut new_state,
                        );
                        if is_theorem.succeeded() {
                            println!("theorem: {:?}", s);
                            state.theorems.extend(tmp_assumptions);
                            state.theorems.push(s.clone());
                        } else {
                            state.bad_theorems.insert(s.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        if res {
            match s {
                PCPSequence::Exact(_) => {}
                _ => {
                    if !theorem_checking && !state.bad_theorems.contains(s) {
                        let mut tmp_assumptions: Vec<PCPSequence> = vec![];
                        let mut new_state = AbstractDFSState {
                            pcp: state.pcp,
                            theorems: state.theorems.clone(),
                            bad_theorems: state.bad_theorems.clone(),
                            emptied: state.emptied.clone(),
                            failed_theorems: state.failed_theorems.clone(),
                        };
                        assumptions.truncate(sofar_count);
                        let is_theorem = check_recursive(
                            s,
                            &mut tmp_assumptions,
                            20,
                            true,
                            false,
                            &mut new_state,
                        );
                        if is_theorem.succeeded() {
                            println!("theorem: {:?}", s);
                            state.theorems.extend(tmp_assumptions);
                            state.theorems.push(s.clone());
                        } else {
                            state.bad_theorems.insert(s.clone());
                        }
                    }
                }
            }
            return TimeoutResult::Result(true);
        }

        assert!(assumptions.len() >= sofar_count);

        assumptions.remove(sofar_count);
        loop {
            let mut removed = false;
            for i in (sofar_count..assumptions.len()).rev() {
                let inexts = assumptions[i].apply_pcp(&state.pcp);
                if !inexts
                    .into_iter()
                    .all(|n| assumptions.iter().any(|s| s.contains(&n)))
                {
                    removed = true;
                    assumptions.remove(i);
                }
            }
            if !removed {
                break;
            }
        }
        //assumptions.truncate(sofar_count);
        //println!("sofar: {sofar_count} assumptions: {:?}", assumptions.len());
    }

    if non_abstracted_empty {
        TimeoutResult::Result(false)
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

fn pdr_like(pcp: &PCP) -> (bool, Vec<Result>) {
    log::debug!("pcp: {:?}", pcp);
    let firsts = ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    }
    .apply_pcp(&pcp);
    log::debug!("firsts: {:?}", firsts);

    let mut state = AbstractDFSState::new(pcp);

    let results = firsts
        .iter()
        .map(|s| -> Result {
            for max_depth_log in 1..5 {
                let max_depth = 1 << max_depth_log;
                println!("max_depth: {:?}", max_depth);
                let mut assumptions = vec![];
                let r = check_recursive(
                    &PCPSequence::Exact(s.clone()),
                    &mut assumptions,
                    max_depth,
                    false,
                    true,
                    &mut state,
                );
                println!("start: {:?} result: {:?}", s.seq, r.succeeded());
                state.theorems.append(&mut assumptions);
                if r.succeeded() {
                    return Result {
                        assumptions: *state.theorems.clone(),
                        result: r.succeeded(),
                        start: PCPSequence::Exact(s.clone()),
                    };
                }
            }
            Result {
                assumptions: *state.theorems.clone(),
                result: false,
                start: PCPSequence::Exact(s.clone()),
            }
        })
        .collect_vec();

    let res = results.iter().all(|s| -> bool { s.result });
    println!("final result: {res} {:?}", pcp);
    (res, results)
}

fn enumerate_substrings_from_pcp(pcp: &PCP, max_len: usize) -> (Vec<String>, Vec<String>) {
    let mut q = VecDeque::new();
    q.push_back(ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    });
    let mut visited = vec![];

    for _ in 0..1000 {
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
            upper_substrs.append(&mut substrings(&s.seq, max_len));
        } else {
            lower_substrs.append(&mut substrings(&s.seq, max_len));
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
        let pcp = parse_pcp_string(&record.as_str());
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
            10000,
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
            10000,
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
    let pcp = parse_pcp_string(&input_str);
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

fn apply_pdr_single(idx: usize, pcp: &PCP, raw: &str, rev: bool) {
    let mut pcp = pcp;
    let reversed = pcp.reverse_pcp().swap_pcp();
    if rev {
        pcp = &reversed;
    }

    println!("{idx} reversed:{rev} pcp: {:?}", raw);
    let (res, results) = pdr_like(&pcp);
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
    for (idx, (raw, pcp)) in parse_file("a.csv").into_iter().enumerate().skip(2) {
        let pcp = pcp;
        apply_pdr_single(idx, &pcp, &raw, false);
        apply_pdr_single(idx, &pcp, &raw, true);
    }
}

// fn check_recursive_bounded(pcp: &PCP, s: &str, dir: PCPDir) -> bool {
//     let mut q = VecDeque::new();

//     q.push_back(PCPSequence::MidWild(MidWildSequence {
//         front: s.to_string(),
//         back: "".to_string(),
//         dir: dir,
//     }));

//     //println!("q: {:?}", q);

//     let mut visited: HashSet<PCPSequence> = HashSet::new();

//     let mut dot_string = String::new();
//     dot_string += "digraph {\n";
//     let mut reced = false;

//     while visited.len() < 1000 && q.len() > 0 {
//         let seq = q.pop_front().unwrap();

//         if visited.contains(&seq) {
//             continue;
//         }
//         visited.insert(seq.clone());

//         let next = seq.apply_pcp(pcp);
//         //println!("{:?} -> {:?}", seq, next);
//         dot_string +=
//             ("\"".to_owned() + &format!("{:?}", seq).replace("\"", "") + "\" -> {").as_str();
//         for n in &next {
//             dot_string += ("\"".to_owned() + &format!("{:?}", n).replace("\"", "") + "\"").as_str();
//         }
//         dot_string += "}\n";

//         for n in next {
//             if n.dir() == dir && n.contains_str(s) {
//                 reced = true;
//                 continue;
//             }
//             if visited.contains(&n) {
//                 continue;
//             }
//             if match &n {
//                 PCPSequence::Exact(e) => e.seq.len() == 0,
//                 _ => false,
//             } {
//                 visited.insert(n.clone());
//                 break;
//             };
//             q.push_back(n);
//         }
//     }

//     dot_string += "}\n";
//     //std::fs::write("graph.dot", dot_string).unwrap();

//     let unreached_empty = visited.iter().all(|s| match s {
//         PCPSequence::Exact(e) => e.seq.len() != 0,
//         _ => true,
//     });

//     //println!("unreached_empty: {unreached_empty}");
//     q.len() == 0 && unreached_empty && reced
// }

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
    for p in result.assumptions.iter() {
        let nexts = p.apply_pcp(pcp);
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
    for pcp in pcps.iter() {
        println!("pcp: {:?}", pcp);
        let (res, results) = pdr_like(pcp);
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
    let pcp = &parse_pcp_string("PCP(Vector(Tile(100,1), Tile(0,100), Tile(1,00)))");
    println!("pcp: {:?}", pcp);
    let (res, results) = pdr_like(pcp);
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

fn main() {
    //check_sanity();
    let pcps = parse_file("a.csv");
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
    //apply_pdr();

    for (idx, (raw, pcp)) in pcps.iter().enumerate().skip(45) {
        println!("{idx} pcp: {:?}", pcp);
        let (us, ds) = find_nonempty_substrs(&pcp);
        let mut ss = us
            .iter()
            .map(|s| {
                PCPSequence::MidExact(MidExactSequence {
                    mid: s.clone(),
                    dir: PCPDir::UP,
                })
            })
            .chain(ds.iter().map(|s| {
                PCPSequence::MidExact(MidExactSequence {
                    mid: s.clone(),
                    dir: PCPDir::DN,
                })
            }))
            .collect_vec();

        // let firsts = ExactSequence {
        //         seq: "".to_string(),
        //         dir: PCPDir::UP,
        //     }
        //     .apply_pcp(&pcp);
        // ss.extend(firsts.iter().map(|s| PCPSequence::Exact(s.clone())));

        let mut state = AbstractDFSState::new(&pcp);

        loop {
            let mut allok = true;
            for (idx, s) in ss.iter().enumerate() {
                let mut others = vec![];
                for (i, o) in ss.iter().enumerate() {
                    if i != idx {
                        others.push(o.clone());
                    }
                }
                println!("{:?} vs {:?}", s, others);
                for max_depth_log in 0..10 {
                    let max_depth = 1 << max_depth_log;
                    let res = check_recursive(&s, &mut others, max_depth, false, false, &mut state);
                    println!("{:?}", res);
                    match res {
                        TimeoutResult::Timeout => {
                            println!("timeout");
                            continue;
                        }
                        TimeoutResult::Result(r) => {
                            if !r {
                                allok = false;
                                break;
                            } else {
                                break;
                            }
                        }
                    }
                }
                if !allok {
                    println!("failed: {:?}", s);
                    state.emptied.push(s.clone());
                    ss.remove(idx);
                    break;
                }
            }
            if allok {
                println!("recursive strings: {:?}", ss);
                println!("all ok");
                break;
            }
        }
    }
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
