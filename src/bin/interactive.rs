use pcp_rec_str::automaton;
use pcp_rec_str::automaton::NFA;
use pcp_rec_str::conf_automaton::ConfAutomaton;
use pcp_rec_str::pcp;
use pcp_rec_str::union_pdr2;

fn read_regex() -> automaton::AppRegex {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    automaton::AppRegex::parse(input.trim())
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let pcp = pcp::PCP::parse_pcp_string(input.trim());
    let mut aut = pcp.to_automaton();

    let mut store: Vec<ConfAutomaton> = vec![];
    loop {
        let upper = read_regex();
        let lower = read_regex();

        store.push(ConfAutomaton {
            upper: NFA::from_regex(&upper),
            lower: NFA::from_regex(&lower),
        });
    }
}
