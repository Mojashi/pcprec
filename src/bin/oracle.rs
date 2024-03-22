use itertools::Itertools;
use pcp_rec_str::automaton::{AppRegex, NFA};
use pcp_rec_str::conf_automaton::{ConfAutomaton, PCPConf};
use std::io::BufRead;

use pcp_rec_str::pcp::{PCPDir, PCP};
use pcp_rec_str::union_pdr2::union_pdr;
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

fn main() {
    //let mut input = String::new();
    //std::io::stdin().read_line(&mut input).unwrap();
    //up=1111&up=1&up=0&dn=1&dn=1101&dn=11
    //let pcp = PCP::parse_pcp_string(input.trim());
    let pcp = PCP::parse_pcp_string("PCP(Vector(Tile(1111,1), Tile(1,1101), Tile(0,11)))");

    let up = PCPConf{
        dir: PCPDir::UP,
        conf: NFA::from_regex(&AppRegex::Or(
            Box::new(AppRegex::parse("((0|1)*00(0|1)*)")), 
            Box::new(AppRegex::Or(
                Box::new(AppRegex::parse("((1*)(0111(1*))*(0(1*))?)")),
                Box::new(AppRegex::parse("((1*)(0111(1*))*(0?))")),
            ))
        )),
        exact: None,
    };
    let dn = PCPConf{
        dir: PCPDir::DN,
        conf: NFA::from_regex(&AppRegex::parse("(1*)(0111(1*))*(011*)?")),
        exact: None
    };

    up.conf.show_dot("up");
    dn.conf.show_dot("dn");

    let nexts = [up.apply_pcp(&pcp), dn.apply_pcp(&pcp)].concat();

    for next in nexts {
        if next.dir == PCPDir::UP {
            println!("{:?}", next);

            if !up.conf.includes(&next.conf) {
                let diff = next.conf.difference(&up.conf);
                let sample = diff.get_element();
                
                println!("{:?}", sample);
                assert!(false)
            }
        } else {
            if !dn.conf.includes(&next.conf) {
                let diff = next.conf.difference(&dn.conf);
                let sample = diff.get_element();
                
                println!("{:?}", sample);
                assert!(false)
            }
        }
    }
}
