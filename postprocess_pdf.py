import os
import pandas as pd
from langchain.text_splitter import MarkdownHeaderTextSplitter
import re
from gemini_api import make_prompt1, make_prompt2, send_request
from dotenv import load_dotenv
import json

from md_split import Splitter

load_dotenv(".env")

def extract_last_itemize_block(text):
        itemize_blocks = re.findall(r'\\begin{itemize}(.*?)\\end{itemize}', text, re.DOTALL)
        if not itemize_blocks:
            return [] 
        last_block = itemize_blocks[-1]
        items = re.findall(r'\\item\s+(.*)', last_block)
        return '\n'.join(items)

class Cleaner:
    def __init__(self):
        self.md_path = os.getenv("MD_FOLDER")
        self.csv_path = os.getenv("CSV_FOLDER")
        self.prompt_path = os.getenv("JSON_FOLDER")
        # self.li = s.ft_generate_splits(self.md_path)
        self.li = []
        with open(self.prompt_path, 'r') as file:
            data = json.load(file)

        self.sys_instruct = r"{}".format(data["Cleaner"]["sys_instruct"])
        self.example_inp_1 = r"{}".format(data["Cleaner"]["example_inp_1"])
        self.example_ctx_1 = r"{}".format(data["Cleaner"]["example_ctx_1"])
        self.example_out_1 = r"{}".format(data["Cleaner"]["example_out_1"])
        self.example_inp_2 = r"{}".format(data["Cleaner"]["example_inp_2"])
        self.example_ctx_2 = r"{}".format(data["Cleaner"]["example_ctx_2"])
        self.example_out_2 = r"{}".format(data["Cleaner"]["example_out_2"])

    def clean_text(self, text):
        cleaned_text = text.replace("\\", "").replace("{", "").replace("}", "")
        return cleaned_text

    def strip_latex_commands(self, s):
        latex_replacements = {
            r'\sin': 'sin',
            r'\cos': 'cos',
            r'\tan': 'tan',
            r'\log': 'log',
            r'\ln': 'ln',
            r'\exp': 'exp',
            r'\alpha': 'alpha',
            r'\beta': 'beta',
            r'\gamma': 'gamma',
            r'\delta': 'delta',
            r'\epsilon': 'epsilon',
            r'\varepsilon': 'epsilon',
            r'\zeta': 'zeta',
            r'\eta': 'eta',
            r'\theta': 'theta',
            r'\vartheta': 'theta',
            r'\iota': 'iota',
            r'\kappa': 'kappa',
            r'\lambda': 'lambda',
            r'\mu': 'mu',
            r'\nu': 'nu',
            r'\xi': 'xi',
            r'\pi': 'pi',
            r'\rho': 'rho',
            r'\sigma': 'sigma',
            r'\tau': 'tau',
            r'\upsilon': 'upsilon',
            r'\phi': 'phi',
            r'\varphi': 'phi',
            r'\chi': 'chi',
            r'\psi': 'psi',
            r'\omega': 'omega',
            r'\hat': '',
            r'\tilde': '',
            r'\bar': '',
            r'\overline': '',
            r'\underline': '',
            r'\mathrm': '',
            r'\mathbf': '',
            r'\mathit': '',
            r'\text': '',
            r'\frac': '/',
            r'\sqrt': 'sqrt',
            r'\left': '',
            r'\right': '',
            r'\bigg': '',
            r'\big': '',
            r'\cdot': '*',
            r'\times': 'x',
            r'\partial': '∂',
            r'\nabla': '∇',
            r'\infty': 'infinity',
            r'\sum': 'sum',
            r'\int': 'integral',
            r'\leq': '<=',
            r'\geq': '>=',
            r'\neq': '!=',
            r'\approx': '~',
            r'\equiv': '≡',
            r'\ldots': '...',
            r'\dots': '...',
            r'\prime': "'",
        }

        for latex_cmd, replacement in latex_replacements.items():
            s = s.replace(latex_cmd, replacement)

        s = re.sub(r'/\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', s)

        s = re.sub(r'\^\{(.*?)\}', r'^\1', s)
        s = re.sub(r'\_\{(.*?)\}', r'_\1', s)
        s = re.sub(r'\^([^\s\^\_])', r'^\1', s)
        s = re.sub(r'\_([^\s\^\_])', r'_\1', s)
        s = s.replace('\\', '')
        s = s.replace('{', '').replace('}', '')
        s = re.sub(r'\s+', ' ', s)
        pattern = re.compile(r'\\[a-zA-Z]+\{(.*?)\}')
        while pattern.search(s):
            s = pattern.sub(r'\1', s)
        s = s.replace('\\', '')

        return s.strip()


    def replace_latex_in_text(self, text):
        patterns = [
        r'\\\((.*?)\\\)',          
        r'\$(.*?)\$',             
        r'\$\$(.*?)\$\$',          
        r'\\textbf\{(.*?)\}',      
        r'\\textit\{(.*?)\}',      
        r'\\emph\{(.*?)\}',        
        r'\\underline\{(.*?)\}',  
        r'\\[a-zA-Z]+\{.*?\}',    
    ]

        combined_pattern = r'|'.join('({})'.format(p) for p in patterns)

        def replacer(match):
            for i in range(1, len(match.groups()) + 1):
                group = match.group(i)
                if group:
                    stripped_content = self.strip_latex_commands(group)
                    return stripped_content
            return match.group(0)  

        new_text = re.sub(combined_pattern, replacer, text, flags=re.DOTALL)

        return new_text
    
    def remove_tables_equations(self, text):
        patterns = [
            re.compile(r'\\begin\{array\}|\\\[.*?\\\]', re.DOTALL),
            re.compile(r'\\begin\{table\}.*?\\end\{table\}', re.DOTALL),
            re.compile(r'\\begin\{tabular\}.*?\\end\{tabular\}', re.DOTALL),
            re.compile(r'\\begin\{table\}.*?(\\begin\{tabular\}.*?\\end\{tabular\}).*?\\end\{table\}', re.DOTALL)
        ]

        for pattern in patterns:
            text = pattern.sub('', text)
        return text
    
    def process_string(self, s, func):
        pattern = re.compile(r'\\begin\{array\}|\\\[.*?\\\]', re.DOTALL)
        longtable_pattern1 = re.compile(r'\\begin{table}.*?\\end{table}', re.DOTALL)
        longtable_pattern2 = re.compile(r'\\begin{tabular}.*?\\end{tabular}', re.DOTALL)
        table_tabular_pattern = re.compile(r'\\begin{table}.*?(\\begin{tabular}.*?\\end{tabular}).*?\\end{table}', re.DOTALL)
        patterns = [table_tabular_pattern, longtable_pattern1, longtable_pattern2, pattern]
        matches = []
        for pat in patterns:
            for m in pat.finditer(s):
                matches.append({'start': m.start(), 'end': m.end(), 'text': m.group()})
        matches.sort(key=lambda x: x['start'])
        lines = s.splitlines()
        line_positions = []
        pos = 0
        for line in lines:
            line_positions.append(pos)
            pos += len(line) + 1  

        def find_line_num(position):
            for i in range(len(line_positions)):
                if i + 1 < len(line_positions):
                    if line_positions[i] <= position < line_positions[i + 1]:
                        return i
                else:
                    return i
            return -1

        def remove_inner_tabulars(text):
            wrapped_match = table_tabular_pattern.search(text)
            if wrapped_match:
                return wrapped_match.group(1)  
            return text 
        result = ''
        last_pos = 0

        def is_enclosed_by_any(start, end, enclosing_matches):
            for enclosing in enclosing_matches:
                if enclosing['start'] <= start < enclosing['end']:
                    return True
            return False
        enclosing_matches = []

        for match in matches:
            start = match['start']
            end = match['end']
            match_text = match['text']

            if is_enclosed_by_any(start, end, enclosing_matches):
                continue
            enclosing_matches.append(match)
            result += s[last_pos:start]
            line_num = find_line_num(start)
            preceding_lines = lines[max(0, line_num - 5):line_num]
            succeeding_lines = lines[line_num + 1:line_num + 5]
            context_lines = preceding_lines + succeeding_lines
            context_lines = [

                line for line in context_lines

                if not (pattern.search(line) or longtable_pattern1.search(line) or longtable_pattern2.search(line))

            ]
            match_text = remove_inner_tabulars(match_text)
            for idx in range(len(context_lines)):
                context_lines[idx] = self.replace_latex_in_text(context_lines[idx])
                context_lines[idx] = self.clean_text(context_lines[idx])
                context_lines[idx] = self.remove_tables_equations(context_lines[idx])
            replacement = func(match_text, context_lines)
            result += replacement
            last_pos = end
        result += s[last_pos:]
        return result
    
        
    def process_input(self, l, c):
        c = "\n".join(c)
        c = c.split()
        c = ' '.join(c[:min(len(c),150)])
        user_question = f'Now, interpret mathematical expressions and/or tabular data to provide an explanation that accurately captures the relationships and significance of the variables and data presented in:\n{l}'
        print(f'Question: {user_question}')
        user_context = f'\nGiven Context:\n{c}'
        PROMPT = make_prompt1(self.sys_instruct, user_context, user_question, self.example_ctx_1, self.example_inp_1, self.example_out_1, self.example_ctx_2, self.example_inp_2, self.example_out_2)
        answer = send_request(PROMPT)
        str_ = extract_last_itemize_block(answer)
        return str_
    
    def clean_files(self):
        s = Splitter()
        self.li = s.ft_generate_splits(self.md_path)
        self.li = [fr"{j}" for j in self.li]
        
        for i in range(len(self.li)):
            m = self.process_string(self.li[i], self.process_input)
            self.li[i] =  m 

        df = pd.DataFrame({'text': self.li})
        df.to_csv(self.csv_path)

